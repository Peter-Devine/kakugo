from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.utils import filesafe_parse_name
from utils.lang_codes import lang_two_letter_code_to_name_map, lang_three_letter_code_to_name_map
from huggingface_hub import HfApi
from datasets import load_dataset

def get_generation_method(row):
    prompt_type = row["prompt_type"]
    scenario = row["scenario"]
    topic = row["topic"]
    if prompt_type is not None:
        return "context"
    if scenario is not None:
        return "scenario"
    if topic is not None:
        return "topic"
    else:
        return "translated"

def get_three_letter_lang_code(lang_name):
    if lang_name not in lang_three_letter_code_to_name_map.values():
        raise Exception(f"""
        Unsupported language: {lang_name}.
        Check the spelling - this language should be one of the languages listed in ./utils/lang_codes.py.
        """)
    three_letter_lang_code = {v: k for k, v in lang_three_letter_code_to_name_map.items()}[lang_name]
    return three_letter_lang_code

def get_shortest_letter_lang_code(lang_name, three_letter_lang_code):
    if lang_name in lang_two_letter_code_to_name_map.values():
        short_lang_code = {v: k for k, v in lang_two_letter_code_to_name_map.items()}[lang_name]
    else:
        short_lang_code = three_letter_lang_code
    return short_lang_code

def get_model_name(lang_name, username):
    three_letter_lang_code = get_three_letter_lang_code(lang_name)
    return f"{username}/kakugo-3B-{three_letter_lang_code}"

def get_dataset_name(lang_name, username):
    three_letter_lang_code = get_three_letter_lang_code(lang_name)
    return f"{username}/kakugo-{three_letter_lang_code}"

def get_model_readme_path(lang_name):
    clean_lang_name = filesafe_parse_name(lang_name)
    return f"./upload/{clean_lang_name}_MODEL_README.md"

def get_dataset_readme_path(lang_name):
    clean_lang_name = filesafe_parse_name(lang_name)
    return f"./upload/{clean_lang_name}_DATASET_README.md"

def make_model_readme(lang_name, username):

    with open("./upload/TEMPLATE_MODEL_CARD.md") as f:
        card_text = f.read()

    three_letter_lang_code = get_three_letter_lang_code(lang_name)
    short_lang_code = get_shortest_letter_lang_code(lang_name, three_letter_lang_code)

    dataset_name = get_dataset_name(lang_name, username)
    model_name = get_model_name(lang_name, username)

    card_text = card_text.replace("XXX_LANGUAGE_CODE", short_lang_code)
    card_text = card_text.replace("XXX_LANGUAGE_NAME", lang_name)
    card_text = card_text.replace("XXX_DATASET", dataset_name)
    card_text = card_text.replace("XXX_MODEL_NAME", model_name)

    readme_path = get_model_readme_path(lang_name)
    with open(readme_path, "w") as f:
        f.write(card_text)

    return readme_path

def make_dataset_readme(lang_name, username):

    with open("./upload/TEMPLATE_DATASET_CARD.md") as f:
        card_text = f.read()

    three_letter_lang_code = get_three_letter_lang_code(lang_name)
    short_lang_code = get_shortest_letter_lang_code(lang_name, three_letter_lang_code)

    dataset_name = get_dataset_name(lang_name, username)
    model_name = get_model_name(lang_name, username)

    card_text = card_text.replace("XXX_LANGUAGE_CODE", short_lang_code)
    card_text = card_text.replace("XXX_LANGUAGE_NAME", lang_name)
    card_text = card_text.replace("XXX_DATASET", dataset_name)
    card_text = card_text.replace("XXX_MODEL_NAME", model_name)

    readme_path = get_dataset_readme_path(lang_name)
    with open(readme_path, "w") as f:
        f.write(card_text)

    return readme_path

def upload_model(lang_name, username, is_private):

    clean_lang_name = filesafe_parse_name(lang_name)

    tokenizer = AutoTokenizer.from_pretrained(f"./train/train_outputs/genreas_trans_full_{clean_lang_name}")
    model = AutoModelForCausalLM.from_pretrained(f"./train/train_outputs/genreas_trans_full_{clean_lang_name}", dtype="auto")

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))

    model_name = get_model_name(lang_name, username)

    tokenizer.push_to_hub(model_name, private=is_private)
    model.push_to_hub(model_name, private=is_private)

    readme_path = make_model_readme(lang_name, username)

    HfApi().upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=model_name,
        repo_type="model",
    )

def upload_dataset(lang_name, username, is_private):
    clean_lang_name = filesafe_parse_name(lang_name)

    ds = load_dataset(
        "json",
        data_files=f"./train/train_data/genreas_trans_full_{clean_lang_name}.json",
        split="train"
    )

    ds = ds.map(
        lambda x: {
            "generation_method": get_generation_method(x)
        },
        num_proc=16
    )
    
    ds = ds.select_columns(
        ["generation_method", "prompt_type", "scenario", "topic", "system", "messages"]
    )

    dataset_name = get_dataset_name(lang_name, username)
    ds.push_to_hub(dataset_name, private=is_private)

    readme_path = make_dataset_readme(lang_name, username)

    HfApi().upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
    )
