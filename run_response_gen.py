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

def main(language, config_data, continue_at_batch_submit):

    os.makedirs("./synthetic_data", exist_ok=True)

    # Translate Auxiliary data
    run_translate_gen(language, config_data, continue_at_batch_submit)

    # Generate responses to generated prompts
    run_response_gen(language, config_data, continue_at_batch_submit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--continue_at_batch_submit", default=False, action='store_true')
    parser.add_argument("--llm_provider", type=str, default="vllm")

    args = parser.parse_args()
    language = args.language

    config_data = read_yaml(args.config_path)
    config_data["llm_provider"] = args.llm_provider
    continue_at_batch_submit = args.continue_at_batch_submit

    main(language, config_data, continue_at_batch_submit)