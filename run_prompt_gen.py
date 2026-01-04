from gen.topic_prompt_gen import generate_topic_prompts
from gen.scenario_prompt_gen import generate_practical_prompts
from gen.context_prompt_gen import generate_contextual_prompts
from gen.prompt_improvement import run_prompt_revision
from utils.utils import read_yaml
import argparse
import os

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