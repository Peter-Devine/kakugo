import pandas as pd
import yaml
import os
import json

CONFIG_FOLDER = "./train/train_configs"
os.makedirs(CONFIG_FOLDER, exist_ok=True)

TRAIN_FOLDER = "/workspace/train"
TRAIN_DATA_FOLDER = f"{TRAIN_FOLDER}/train_data"
OUTPUT_FOLDER = f"{TRAIN_FOLDER}/train_outputs"
LOCAL_DATASET_INFO_PATH = "./train/dataset_info.json"

def save_config(config_template, run_id):

    dataset_info = json.load(open(LOCAL_DATASET_INFO_PATH))
    dataset_info[run_id] = {
        "file_name": f"{TRAIN_DATA_FOLDER}/{run_id}.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "systen": "system",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
    
    with open(LOCAL_DATASET_INFO_PATH, "w") as f:
        json.dump(dataset_info, f, indent=4)

    config_template["dataset"] = run_id

    output_path = f"{OUTPUT_FOLDER}/{run_id}"

    config_template["output_dir"] = output_path
    config_template["run_name"] = run_id

    with open(f"{CONFIG_FOLDER}/{run_id}.yaml", 'w') as outfile:
        yaml.safe_dump(config_template, outfile)

def make_train_config(language, config):

    with open("./train/llama_factory_template.yaml") as stream:
        config_template = yaml.safe_load(stream)

    save_config(config_template, f"genreas_trans_full_{language}")
