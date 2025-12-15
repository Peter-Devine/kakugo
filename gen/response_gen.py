from utils.utils import get_responses, filesafe_parse_name, get_batch_responses, SYNTHETIC_DATA_FOLDER
import pandas as pd
from glob import glob
import os

#### Prompt functions ####

def gen_responses(config, df, lang_name, system_prompt_template, data_name, continue_at_batch_submit, has_context=False):

    has_revised_mask = df["revised_prompt"].notna()
    if has_context:
        prompts = [f"{t}\n\n{rp}" if has_rev else f"{t}\n\n{p}" for t, p, rp, has_rev in zip(df["text"], df["prompt"], df["revised_prompt"], has_revised_mask)]
    else:
        prompts = [rp if has_rev else p for p, rp, has_rev in zip(df["prompt"], df["revised_prompt"], has_revised_mask)]

    df["final_prompt"] = prompts

    system_prompts = [system_prompt_template.format(lang_name=lang_name) for _ in prompts]
    df["system_prompt"] = system_prompts
    
    temp_df = df[df["final_prompt"].notna()]
    system_prompts = temp_df["system_prompt"].tolist()
    final_prompts = temp_df["final_prompt"].tolist()

    do_batch = config["do_batch"]
    if do_batch:
        full_responses = get_batch_responses(config, system_prompts, final_prompts, f"responses_{data_name}", continue_at_batch_submit)
        if continue_at_batch_submit:
            return None
    else:
        full_responses = get_responses(config, system_prompts, final_prompts)

    reasonings = [r["reasoning"] for r in full_responses]
    responses = [r["response"] for r in full_responses]

    response_col = "response"
    temp_df[response_col] = responses
    reasoning_col = "reasoning"
    temp_df[reasoning_col] = reasonings

    df = temp_df[[response_col, reasoning_col]].combine_first(df)
    return df

def run_response_gen(language, config, continue_at_batch_submit):

    system_prompt_template = config["response_gen_system_prompt_template"]

    safe_language = filesafe_parse_name(language)
    paths = sorted(glob(f'{SYNTHETIC_DATA_FOLDER}/{safe_language}/*_prompts.jsonl'))

    for path in paths:
        print(f"Generating responses for: {path}")
        has_context = path.split("/")[-1].startswith("contextual_")
        df = pd.read_json(path, lines=True)
        data_name = safe_language + "_" + os.path.basename(path).split(".")[0]
        df = gen_responses(config, df, language, system_prompt_template, data_name, continue_at_batch_submit, has_context=has_context)
        if continue_at_batch_submit:
            continue
        df.to_json(path, lines=True, orient="records")