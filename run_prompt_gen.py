from gen.topic_prompt_gen import generate_topic_prompts
from gen.scenario_prompt_gen import generate_practical_prompts
from gen.context_prompt_gen import generate_contextual_prompts
from gen.prompt_improvement import run_prompt_revision
from gen.response_gen import run_response_gen
from translate.translate_data import run_translate_gen
import argparse
import os
import yaml

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def main(language, config_data):

    os.makedirs("./synthetic_data", exist_ok=True)

    # Generate prompts
    generate_topic_prompts(language, config_data)
    generate_practical_prompts(language, config_data)
    generate_contextual_prompts(language, config_data)
    run_prompt_revision(language, config_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--llm_provider", type=str, default="vllm")

    args = parser.parse_args()
    language = args.language

    config_data = read_yaml(args.config_path)
    config_data["llm_provider"] = args.llm_provider

    main(language, config_data)