from gen.response_gen import run_response_gen
from translate.translate_data import run_translate_gen
from utils.utils import read_yaml
import argparse
import os

def main(language, config_data, continue_at_batch_submit):

    os.makedirs("./synthetic_data", exist_ok=True)

    # Translate Auxiliary translated data
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