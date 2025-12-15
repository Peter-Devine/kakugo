from utils.utils import get_responses, get_batch_responses, filesafe_parse_name, TRANSLATED_FOLDER, TRANSLATED_RAW_ENG_FOLDER
from datasets import load_dataset
import os
import ast
import json
from transformers import AutoTokenizer

def get_system_prompt(lang_name):
    return f"""You are an English to {lang_name} translation assistant.
  Given a list of dicts that form a conversation in English between a user and assistant, output the {lang_name} translation of that conversation in the same list of dicts format.
  Only include this list of dicts in your final output.
  Keep the keys and format of the list of dicts the same in your translation.
"""

def parse_messages(translated_messages):
    try:
        return json.dumps(ast.literal_eval(translated_messages))
    except:
        return None

def run_translate_gen(language, config, continue_at_batch_submit):

    translation_number_rows = config["translation_number_rows"]
    translation_max_input_tokens = config["translation_max_input_tokens"]
    do_batch = config["do_batch"]

    os.makedirs(TRANSLATED_RAW_ENG_FOLDER, exist_ok=True)
    raw_english_path = os.path.join(TRANSLATED_RAW_ENG_FOLDER, f"{language}.json")
    if os.path.isfile(raw_english_path):
        ds = load_dataset("json", data_files=raw_english_path, split="train")
    else:
        ds = load_dataset("BAAI/Infinity-Instruct", "7M_core", split="train").filter(
            lambda x: x["langdetect"] == "en",
            num_proc=16
        )
        ds = ds.shuffle().select(range(translation_number_rows))
        ds = ds.to_json(raw_english_path)

    role_map = {"human": "user", "gpt": "assistant"}
    ds = ds.map(
        lambda x: {
            "messages": [
                {
                    "role": role_map[turn["from"]],
                    "content": turn["value"],
                } for turn in x["conversations"]]
        }
    )

    tokenizer_name = config["synthetic_generation_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def is_correct_len(messages):
        return len(tokenizer.encode(str(messages))) <= translation_max_input_tokens

    ds = ds.filter(
        lambda x: is_correct_len(x["messages"]),
        num_proc=16
    )

    prompts = [str(x) for x in ds["messages"]]
    system_prompts = [get_system_prompt(language)] * len(prompts)

    print(f"Translating {language}")
    if do_batch:
        outputs = get_batch_responses(config, system_prompts, prompts, f"translate_{language}", continue_at_batch_submit)
        if continue_at_batch_submit:
            return None
    else:
        outputs = get_responses(config, system_prompts, prompts)

    outputs = [o["response"] for o in outputs]

    ds = ds.add_column("translated_messages_str", outputs)

    ds = ds.map(
        lambda x: {"translated_messages": parse_messages(x["translated_messages_str"])},
        num_proc=16
    )

    os.makedirs(TRANSLATED_FOLDER, exist_ok=True)
    ds.to_json(
        f"{TRANSLATED_FOLDER}/{filesafe_parse_name(language)}.json"
    )

