from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import os
import json
from utils.utils import filesafe_parse_name
from transformers import AutoTokenizer
import numpy as np

TRAIN_FOLDER = "./train/train_data/"
os.makedirs(TRAIN_FOLDER, exist_ok=True)

def has_utf_errors(text):
    try:
        text.encode('utf-8', errors='strict')
        return False
    except:
        return True

def is_correct_format(conversation):
    required_keys = set(["role", "content"])
    roles = set(["user", "assistant"])
    previous_turn = "assistant"
    for turn in conversation:
        if not isinstance(turn, dict):
            return False
        if len(turn.keys()) != 2:
            return False
        elif len(set(turn.keys()) - required_keys) > 0:
            return False
        elif turn["role"] not in roles:
            return False
        elif turn["role"] == previous_turn:
            return False
        elif has_utf_errors(turn["content"]):
            return False
        else:
            previous_turn = turn["role"]
    return True

def get_generated_train_data(language, config, subset):

    tokenizer_name = config["train_model_tokenizer_name"]

    dataset_path = f"./synthetic_data/{language}/{subset}_prompts.jsonl"

    if not os.path.isfile(dataset_path):
        if subset == "contextual":
            return Dataset.from_pandas(pd.DataFrame({"messages": [], "num_tokens": []}))
        else:
            raise Exception(f"Missing dataset for {dataset_path}")

    # In general, Pandas seems to be more robust with loading potentially noisy .jsonl than Datasets
    df = pd.read_json(dataset_path, lines=True).map(str)

    ds = Dataset.from_pandas(
        df
    )

    ds = ds.filter(
        lambda x: isinstance(x["final_prompt"], str) and isinstance(x["response"], str),
        num_proc=16
    )

    ds = ds.map(
        lambda x: {"messages": [
            {"role": "user", "content": x["final_prompt"]},
            {"role": "assistant", "content": x["response"]},
        ]},
        num_proc=16
    )

    bad_tokens = ["<image>", "<video>", "<audio>"]
    ds = ds.filter(
        lambda x: not any([any(bt in turn["content"] for bt in bad_tokens) for turn in x["messages"]]),
        num_proc=16
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = ds.map(
        lambda x: {"num_tokens": len(tokenizer.apply_chat_template(x["messages"], tokenize=True))},
        num_proc=16
    )    
    
    return ds.shuffle()

def is_tok_diff_coeff_in_limits(tokenizer, original_messages, translated_messages, max_coeff, min_coeff):
    total_tokens = 0
    for original_message, translated_message in zip(original_messages, translated_messages):
        num_trans_toks = len(tokenizer.encode(translated_message["content"]))
        total_tokens += num_trans_toks
        num_orig_toks = len(tokenizer.encode(original_message["content"]))
        coeff = float(num_trans_toks / num_orig_toks)
        if bool(coeff < min_coeff) or bool(coeff > max_coeff):
            return False
    return True

def is_json_parseable(text):
    try:
        json.loads(text)
        return True
    except:
        return False

def get_translated_train_data(language, config):

    min_coeff = config["translation_min_len_coeff"]
    max_coeff = config["translation_max_len_coeff"]
    tokenizer_name = config["train_model_tokenizer_name"]

    translation_path = f"./translated_data/{filesafe_parse_name(language)}.json"

    ds = load_dataset(
        "json",
        data_files=translation_path, 
        split="train"
    )
    
    ds = ds.filter(
        lambda x: is_json_parseable(x["translated_messages"]) if x["translated_messages"] else False
    )

    ds = ds.filter(
        lambda x: is_correct_format(json.loads(x["translated_messages"]))
    )

    ds = ds.map(
        lambda x: {
            "translated_messages": json.loads(
                x["translated_messages"]
            )
        }
    )

    ds = ds.filter(
        lambda x: len(x["messages"]) == len(x["translated_messages"]) # Filter out translations that do not match the original number of turns
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = ds.filter(
        lambda x:
            is_tok_diff_coeff_in_limits(
                tokenizer, 
                x["messages"], 
                x["translated_messages"], 
                max_coeff, 
                min_coeff,
        )
    )

    if "messages" in ds.features:
        ds = ds.remove_columns(["messages"])

    ds = ds.rename_column("translated_messages", "messages")

    return ds

def make_gen_trans_full_train_data(language, config):
    trans_ds = get_translated_train_data(language, config)
    cont_ds = get_generated_train_data(language, config, "contextual")
    scen_ds = get_generated_train_data(language, config, "scenario")
    topic_ds = get_generated_train_data(language, config, "topic")
    gen_ds = concatenate_datasets([cont_ds, scen_ds, topic_ds]).shuffle()

    think_start_string = config["train_model_think_start_string"]
    think_end_string = config["train_model_think_end_string"]

    # Add reasoning traces to generated data
    gen_reas_ds = gen_ds.map(
        lambda x: {
            "messages": [
                {"role": "user", "content": x["messages"][0]["content"]},
                {"role": "assistant", "content": think_start_string + " " + x["reasoning"] + " " + think_end_string + " " + x["messages"][1]["content"]},
            ]
        }
    )

    # Add system messages
    reas_sys_prompt = f"Before you respond, first think about your response and enclose your thinking process in {think_start_string} and {think_end_string} delimiters."
    gen_reas_ds = gen_reas_ds.add_column(
        "system", [reas_sys_prompt] * len(gen_reas_ds)
    )
    trans_ds = trans_ds.add_column(
        "system", ["Be concise in your responses."] * len(trans_ds)
    )

    concatenate_datasets([gen_reas_ds, trans_ds]).shuffle().to_json(
        TRAIN_FOLDER + f"genreas_trans_full_{language}.json"
    )