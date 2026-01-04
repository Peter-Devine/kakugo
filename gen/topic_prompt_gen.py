import json
from tqdm.auto import tqdm
import os
import pandas as pd
from utils.utils import get_responses, parse_response_json, to_jsonl, filesafe_parse_name, SYNTHETIC_DATA_FOLDER

def get_general_seed_topics():
    return [ 
        "daily life",
        "the world",
        "health",
        "practical skills",
        "arts and culture",
        "sciences",
        "social sciences",
        "humanities",
    ]

def get_target_seed_topics(language_name):
    return [
        f"{language_name} speaking places",
        f"{language_name} speaking people",
        f"{language_name} culture",
        f"{language_name} language",
        f"{language_name} history",
        f"{language_name} society",
        f"{language_name} daily life",
        f"{language_name} health",
    ]

#### Prompt functions ####

def get_topics_gen_prompt(macrotopic, num_topics, lang_name, min_topics = 1):

    system_prompt = """You are a topic generating assistant. Your topic names should be as concise as possible.
    Your output should only consist of a single JSON object within a JSON block (delimited by ```json and ```) which has one key, "topics", which is an array of topic strings.
    Do not include asterisks (*) anywhere within your JSON."""

    lang_specific_prompt = f"""\n\nYour topics should be relevant to people who speak {lang_name}.""" if lang_name is not None else ""

    prompt = f"""Write {num_topics} topics, things, people, places, objects, concepts, themes, or anything else that encompass the main aspects of {macrotopic}.
    Your topics should be diverse, comprehensive, concise, and be written in simple English.""" + lang_specific_prompt

    return system_prompt, prompt

def get_prompt_gen_prompt(prompt_gen_template, topic, lang_name, num_prompts, is_problem, min_prompts=1):

    output_type = "problem" if is_problem else "prompt"

    system_prompt = prompt_gen_template.format(lang_name=lang_name)

    prompt = f"""Write at least {min_prompts} and at most {num_prompts} {lang_name} {output_type}s related to {topic}"""

    if is_problem:
        prompt += "."
    else:
        prompt += f" that a user would ask an AI assistant."

    return system_prompt, prompt

def generate_topics(config, macrotopics, num_topics, lang_name=None):
    prompt_dict = {}
    for macrotopic in tqdm(macrotopics):
        system_prompt, prompt = get_topics_gen_prompt(macrotopic, num_topics=num_topics, lang_name=lang_name)
        prompt_dict[macrotopic] = (system_prompt, prompt)
        
    system_prompts = [prompt_dict[macrotopic][0] for macrotopic in macrotopics]
    prompts = [prompt_dict[macrotopic][1] for macrotopic in macrotopics]

    outputs = get_responses(config, system_prompts, prompts)
    outputs = [parse_response_json(output["response"]) for output in outputs]
    topics = []
    for macrotopic, output in zip(macrotopics, outputs):
        if isinstance(output, dict) and "topics" in output:
            for topic in output["topics"]:
                if not isinstance(topic, str):
                    continue
                topics.append({"macrotopic": macrotopic, "topic": topic})

    lang_name = "General" if lang_name is None else lang_name
    return [x["topic"] for x in topics]

def generate_prompts(config, topic_list, lang_name, num_prompts, prompt_gen_template, is_problem=False):
    prompt_data = []
    for topic in topic_list:
        system_prompt, prompt = get_prompt_gen_prompt(prompt_gen_template, topic, lang_name, num_prompts=num_prompts, is_problem=is_problem)
        prompt_data.append(
            {
                "topic": topic,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "lang_name": lang_name,
            }
        )

    system_prompts = [x["system_prompt"] for x in prompt_data]
    prompts = [x["prompt"] for x in prompt_data]

    outputs = get_responses(config, system_prompts, prompts)
    prompts = [parse_response_json(output["response"]) for output in outputs]
    prompts = [x.get("prompts") if isinstance(x, dict) else None for x in prompts]

    prompt_df = pd.DataFrame(prompt_data)[["topic"]]
    prompt_df["prompt"] = prompts
    prompt_df = prompt_df.explode("prompt")
    prompt_df = prompt_df.reset_index(drop=False)

    lang_prompt_folder = f"{SYNTHETIC_DATA_FOLDER}/{filesafe_parse_name(lang_name)}"
    os.makedirs(lang_prompt_folder, exist_ok=True)
    to_jsonl(prompt_df, f"{lang_prompt_folder}/topic_prompts.jsonl")
    return prompt_df

def generate_topic_prompts(language, config):

    num_general_macrotopics = config["open_num_general_macrotopics"]
    num_target_macrotopics = config["open_num_target_macrotopics"]
    num_general_topics = config["open_num_general_topics"]
    num_target_topics = config["open_num_target_topics"]
    prompts_per_topic = config["open_prompts_per_topic"]
    prompt_gen_template = config["prompt_gen_system_prompt_template"]

    # Get seed topics
    general_seed_topics = get_general_seed_topics()
    target_seed_topics = get_target_seed_topics(language)

    # Get macrotopics from seed topics
    general_macrotopics = generate_topics(config, general_seed_topics, num_general_macrotopics)
    target_macrotopics = generate_topics(config, target_seed_topics, num_target_macrotopics, lang_name=language)
    
    # Get topics from macrotopics
    general_topics = generate_topics(config, general_macrotopics, num_general_topics)
    target_topics = generate_topics(config, target_macrotopics, num_target_topics, lang_name=language)

    topics = general_topics + target_topics + general_macrotopics + target_macrotopics + general_seed_topics + target_seed_topics
    topics = list(set(topics))

    # Generate prompts from topics
    generate_prompts(config, topics, language, prompts_per_topic, prompt_gen_template)