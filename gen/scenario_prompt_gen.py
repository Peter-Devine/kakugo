import json
import os
from utils.utils import get_responses, parse_response_json
import pandas as pd
from utils.utils import filesafe_parse_name, to_jsonl, SYNTHETIC_DATA_FOLDER

#### Prompt functions ####

def get_scenario_gen_prompt(num_scenarios, general_scenario=None, lang_name=None, min_scenarios=1):
    system_prompt = """You are a scenario generating assistant. Your scenario names should be as concise as possible.
    Your output should only consist of a single JSON object within a JSON block (delimited by ```json and ```) which has one key, "scenarios", which is an array of scenario strings."""
    
    lang_specific_prompt = f"""\n\nConsider the scenarios in which a {lang_name} speaker would use an AI assistant.
    Your scenarios should be specific to users who speak {lang_name}.""" if lang_name is not None else ""
    
    seeded_instruction = f"Write {num_scenarios} specific practical scenarios that people would use an AI assistant for when they are using the assistant for the following task: {general_scenario}."
    unseeded_instruction = f"Write the {num_scenarios} most popular scenarios that people would use an AI chatbot for."
    instruction = unseeded_instruction if general_scenario is None else seeded_instruction
    
    prompt = instruction + f"""
    
    Do not include asterisks (*) in any of your output.

    Your scenarios should be general and concise, rather than detailed.
    Your scenarios should be diverse, realistic, practical, comprehensive, concise, and be written in English.""" + lang_specific_prompt
    return system_prompt, prompt

def get_prompt_gen_prompt(prompt_gen_template, scenario, lang_name, n_prompts):
    system_prompt = prompt_gen_template.format(lang_name=lang_name)
    prompt = f"""Write {n_prompts} prompts in {lang_name} that a user would ask an AI chatbot when they are using the chatbot in the following scenario: {scenario}."""
    return system_prompt, prompt

def generate_scenarios(config, num_scenarios, broad_scenarios=None, lang_name=None):
    if broad_scenarios is not None:
        sysprompts = [get_scenario_gen_prompt(general_scenario=x, num_scenarios=num_scenarios, lang_name=lang_name) for x in broad_scenarios]
    else:
        sysprompts = [get_scenario_gen_prompt(num_scenarios=num_scenarios, lang_name=lang_name)]
    system_prompts = [x[0] for x in sysprompts]
    prompts = [x[1] for x in sysprompts]

    scenarios_responses = get_responses(config, system_prompts, prompts)
    scenarios_responses = [parse_response_json(response["response"]) for response in scenarios_responses]

    if broad_scenarios is None:
        # If we get an error when generating broad scenarios, we retry until we get a correct output.
        # We need the broad scenarios to generate detailed scenarios, so if we miss this, then we lose a substantial amount of data.
        while scenarios_responses[0] is None:
            print("Retrying broad scenario generation")
            scenarios_responses = get_responses(config, system_prompts, prompts)
            scenarios_responses = [parse_response_json(response["response"]) for response in scenarios_responses]

    scenarios_list = []
    for s in scenarios_responses:
        if isinstance(s, dict) and "scenarios" in s:
            scenarios_list.extend(s["scenarios"])
    return scenarios_list

def generate_prompts(config, scenario_list, n_prompts, prompt_gen_template, lang_name):
    prompt_data = []
    for scenario in scenario_list:
        system_prompt, prompt = get_prompt_gen_prompt(prompt_gen_template, scenario, lang_name=lang_name, n_prompts=n_prompts)
        prompt_data.append(
            {
                "scenario": scenario,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "lang_name": lang_name,
            }
        )
    system_prompts = [x["system_prompt"] for x in prompt_data]
    prompts = [x["prompt"] for x in prompt_data]
    prompts_responses = get_responses(config, system_prompts, prompts)
    prompts_responses = [parse_response_json(response["response"]) for response in prompts_responses]
    prompts = [response["prompts"] if isinstance(response, dict) and "prompts" in response else None for response in prompts_responses]
    
    prompt_df = pd.DataFrame(prompt_data)[["scenario"]]
    prompt_df["prompt"] = prompts
    prompt_df = prompt_df.explode("prompt")
    prompt_df = prompt_df.reset_index(drop=False)

    lang_prompt_folder = f"{SYNTHETIC_DATA_FOLDER}/{filesafe_parse_name(lang_name)}"
    os.makedirs(lang_prompt_folder, exist_ok=True)
    to_jsonl(prompt_df, f"{lang_prompt_folder}/scenario_prompts.jsonl")
    return prompt_df

def generate_practical_prompts(language, config):

    num_broad_scenarios = config["scenario_num_broad_scenarios"]
    num_detailed_scenarios = config["scenario_num_detailed_scenarios"]
    prompts_per_topic = config["scenario_prompts_per_topic"]
    prompt_gen_template = config["prompt_gen_system_prompt_template"]

    # Get broad scenarios from no seed
    general_broad_scenarios = generate_scenarios(config, num_broad_scenarios)
    target_broad_scenarios = generate_scenarios(config, num_broad_scenarios, lang_name=language)

    # Get detailed scenarios from broad scenarios
    general_detailed_scenarios = generate_scenarios(config, num_detailed_scenarios, broad_scenarios=general_broad_scenarios)
    target_detailed_scenarios = generate_scenarios(config, num_detailed_scenarios, broad_scenarios=target_broad_scenarios, lang_name=language)

    scenarios = general_detailed_scenarios + target_detailed_scenarios + general_broad_scenarios + target_broad_scenarios

    # Get prompts from detailed scenarios
    generate_prompts(config, scenarios, prompts_per_topic, prompt_gen_template, lang_name=language)