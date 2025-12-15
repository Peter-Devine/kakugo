from tqdm.auto import tqdm
from utils.utils import get_responses, parse_response_json
import pandas as pd
from datasets import load_dataset
from utils.lang_codes import lang_script_code_to_name_map
from utils.utils import filesafe_parse_name, SYNTHETIC_DATA_FOLDER
from transformers import AutoTokenizer
import os
import random

name_to_lang_script_code_map = {v: k for k, v in lang_script_code_to_name_map.items()}

#### Prompt functions ####

def get_prompt_gen_prompt(prompt_gen_template, context_text, prompt_type, num_prompts, lang_name):
    system_prompt = prompt_gen_template.format(lang_name=lang_name)
    prompt = f"""Write {num_prompts} prompts in {lang_name} related to the following text:

    {context_text}

    The prompts should be written in natural {lang_name} and should ask the AI assistant to {prompt_type} the text."""
    return system_prompt, prompt

def generate_contextual_prompts(language, config):

    qa_weighting = config["contextual_qa_weighting"]
    num_contexts = config["contextual_num_contexts"]
    first_n_tokens = config["contextual_first_n_tokens"]
    contextual_tokenizer = config["contextual_tokenizer"]
    num_prompts_per_text = config["contextual_num_prompts_per_text"]
    prompt_gen_template = config["prompt_gen_system_prompt_template"]

    prompt_types = [
        "translate",
        "summarize",
        "improve",
        "classify",
    ] +list(["answer a question about"] * qa_weighting)

    if language not in name_to_lang_script_code_map.keys():
        print("Not generating any contextual prompts because language is not supported by fineweb-2")
        return

    ds = load_dataset("HuggingFaceFW/fineweb-2", name_to_lang_script_code_map[language], split="train")
    original_dataset_len = len(ds)
    ds = ds.shuffle().select(range(min(num_contexts, original_dataset_len)))
    texts = ds["text"]
    tokenizer = AutoTokenizer.from_pretrained(contextual_tokenizer)
    texts = [tokenizer.decode(tokenizer.encode(x, add_special_tokens=False)[:first_n_tokens]) for x in texts]

    prompt_data = []
    for i, text in enumerate(tqdm(texts)):

        prompt_type = prompt_types[i % len(prompt_types)]
        system_prompt, prompt = get_prompt_gen_prompt(prompt_gen_template, text, prompt_type, num_prompts_per_text, language)

        prompt_data.append(
            {
                "text": text,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "lang_name": language,
            }
        )

    system_prompts = [x["system_prompt"] for x in prompt_data]
    prompts = [x["prompt"] for x in prompt_data]

    outputs = get_responses(config, system_prompts, prompts)
    prompts = [parse_response_json(output["response"]) for output in outputs]
    prompts = [response["prompts"] if isinstance(response, dict) and "prompts" in response else None for response in prompts]

    prompt_df = pd.DataFrame(prompt_data)[["text", "prompt_type"]]
    prompt_df["prompts"] = prompts
    
    if original_dataset_len > num_contexts:
        prompt_df["prompt"] = prompt_df["prompts"].apply(
            lambda x: random.choice(x) if isinstance(x, list) else None
        )
    else:
        prompt_df["prompt"] = prompt_df["prompts"]
        prompt_df = prompt_df.explode("prompt")
        total_num_prompts = len(prompt_df)
        if num_contexts < total_num_prompts:
            prompt_df = prompt_df.sample(n=num_contexts)
    
    lang_prompt_folder = f"{SYNTHETIC_DATA_FOLDER}/{filesafe_parse_name(language)}"
    os.makedirs(lang_prompt_folder, exist_ok=True)
    prompt_df.to_json(f"{lang_prompt_folder}/contextual_prompts.jsonl", lines=True, orient="records")
    return prompt_df