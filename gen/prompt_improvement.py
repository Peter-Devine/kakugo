from utils.utils import get_responses, parse_response_json
import pandas as pd
from glob import glob
from utils.utils import filesafe_parse_name, SYNTHETIC_DATA_FOLDER

#### Prompt functions ####

def get_improvement_prompt(initial_prompt, lang_name, sys_prompt_template):

    system_prompt = sys_prompt_template.format(lang_name=lang_name)

    prompt = f"""Prompt:
    
    {initial_prompt}"""
    return system_prompt, prompt

def get_context_improvement_prompt(context_text, initial_prompt, lang_name, sys_prompt_template):

    system_prompt = sys_prompt_template.format(lang_name=lang_name)
    system_prompt += "\n\nDo NOT output any of the given context text, only improve the given prompt. However, the prompt should still refer to the context text."

    prompt = f"""Context text:

    {context_text}
    
    Prompt:
    
    {initial_prompt}"""
    return system_prompt, prompt

def revise_prompts(config, df, lang_name, sys_prompt_template, perc_prompt_to_revise, has_context):
    sampled_df = df.sample(frac=perc_prompt_to_revise)
    if has_context:
        prompts = [get_context_improvement_prompt(text, prompt, lang_name, sys_prompt_template) for text, prompt in zip(sampled_df["text"], sampled_df["prompt"])]
    else:
        prompts = [get_improvement_prompt(x, lang_name, sys_prompt_template) for x in sampled_df["prompt"]]

    system_prompts = [x[0] for x in prompts]
    prompts = [x[1] for x in prompts]

    revision_responses = get_responses(config, system_prompts, prompts)
    revision_responses = [parse_response_json(x["response"]) for x in revision_responses]
    revised_prompt_col = "revised_prompt"
    revised_prompts = [x.get("improved_prompt", None) if isinstance(x, dict) else None for x in revision_responses]
    sampled_df[revised_prompt_col] = revised_prompts
    df = sampled_df[[revised_prompt_col]].combine_first(df)
    return df

def run_prompt_revision(language, config):

    paths = glob(f'{SYNTHETIC_DATA_FOLDER}/{filesafe_parse_name(language)}/*_prompts.jsonl')

    sys_prompt_template = config["prompt_improvement_system_prompt_template"]
    perc_prompt_to_revise = config["prompt_improvement_perc_prompt_to_revise"]

    for path in paths:
        print(f"Improving {path}")
        has_context = path.split("/")[-1].startswith("contextual_")
        df = pd.read_json(path, lines=True)
        df = revise_prompts(config, df, language, sys_prompt_template, perc_prompt_to_revise, has_context=has_context)
        df.to_json(path, lines=True, orient="records")