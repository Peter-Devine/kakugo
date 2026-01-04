import yaml
import json
import re
import os
from datasets import Dataset
import os
import time
import yaml

SYNTHETIC_DATA_FOLDER = "./synthetic_data"
TRANSLATED_RAW_ENG_FOLDER = "./translated_english"
TRANSLATED_FOLDER = "./translated_data"
BATCH_INPUT_FOLDER = "./batch_inputs"
BATCH_OUTPUTS_FOLDER = "./batch_outputs"
BATCH_ID_FOLDER = "./batch_ids"

# If the llm_provider is vLLM, then keep the LLM saved in a global dict so it does not need to be reloaded between each inference batch
llm_dict = {}

def parse_json(json_data):
    try:
        return json.loads(json_data)
    except Exception as e:
        print(e)
        print(json_data)
        return None

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

# This replaces invalid characters with a '?' or removes them
# Does this as some scripts such as Tibetan fail to save properly due to 'surrogates not allowed' error.
def clean_surrogates(val):
    if isinstance(val, str):
        return val.encode('utf-8', 'ignore').decode('utf-8')
    return val

def to_jsonl(df, path):
    df = df.map(clean_surrogates)
    df.to_json(path, lines=True, orient="records")

def parse_response_json(response):
    if response is None:
        return None

    if response.strip().lower().startswith("json"):
        response = response.strip()[len("json"):]
    return parse_json(response.split("```json")[-1].replace("```", ""))

def parse_reas_resp(response, thinking_start_token, thinking_end_token):
    if bool(not isinstance(response, str)) or bool(thinking_end_token not in response):
        return {"reasoning": None, "response": None}

    response = response.strip()

    if bool(response.startswith(thinking_start_token)):
        response = response[len(thinking_start_token):].strip()

    reasoning = response.split(thinking_end_token)[0]
    response = response.split(thinking_end_token)[-1]

    return {
        "reasoning": reasoning, "response": response
    }

def get_conversations(system_prompts, prompts):
    return [[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        },
    ] for system_prompt, prompt in zip(system_prompts, prompts)]

def unpack_vllm_response(response_output):
    text_outputs = [o.text if o.finish_reason == "stop" else None for o in response_output.outputs]
    if len(text_outputs) == 1:
        return text_outputs[0]
    else:
        return text_outputs

def get_vllm_responses(llm, sampling_params, system_prompts, prompts, thinking_start_token, thinking_end_token):
    conversations = get_conversations(system_prompts, prompts)
    response_outputs = llm.chat(conversations,
        sampling_params=sampling_params,
        use_tqdm=True
    )
    full_output = [unpack_vllm_response(response_output) for response_output in response_outputs]
    return [parse_reas_resp(x, thinking_start_token, thinking_end_token) for x in full_output]

def get_together_output(together_client, model_name, temperature, messages, max_tokens):
    try:
        outputs = together_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as e:
        print(f"Together call error: {e}")
        return {
            "reasoning": None,
            "response": None
        }
    
    choice = outputs.choices[0]
    message = choice.message
    if choice.finish_reason.value != "stop":
        return {
            "reasoning": None,
            "response": None
        }
    else:
        return {
            "reasoning": message.reasoning,
            "response": message.content
        }

def get_together_responses(together_client, model_name, temperature, num_proc, system_prompts, prompts, max_tokens):
    conversations = get_conversations(system_prompts, prompts)
    ds = Dataset.from_dict(
        {"messages": conversations}
    )
    ds = ds.map(
        lambda x: get_together_output(together_client, model_name, temperature, x["messages"], max_tokens),
        num_proc=num_proc
    )
    return ds.remove_columns(["messages"]).to_list()

def get_together_client():
    from together import Together
    return Together(api_key=os.environ["TOGETHER_AI_KEY"])

def get_responses(config_data, system_prompts, prompts):
    llm_provider = config_data["llm_provider"]

    if llm_provider == "vllm":
        from vllm import LLM, SamplingParams

        if "llm" not in llm_dict.keys():
            llm_dict["llm"] = LLM(
                model=config_data["synthetic_generation_model_name"], 
                tensor_parallel_size=config_data["synthetic_generation_tensor_parallel_size"], 
                trust_remote_code=config_data["synthetic_generation_trust_remote_code"], 
                gpu_memory_utilization=config_data["synthetic_generation_gpu_memory_utilization"], 
                max_model_len=config_data["synthetic_generation_max_model_len"], 
                max_num_seqs=config_data["synthetic_generation_max_num_seqs"], 
            )
            llm_dict["sampling_params"] = SamplingParams(
                temperature=config_data["synthetic_generation_temperature"],
                max_tokens=config_data["synthetic_generation_max_model_len"],
            )
        llm = llm_dict["llm"]
        sampling_params = llm_dict["sampling_params"]

        thinking_start_token = config_data.get("synthetic_generation_model_think_start_string", None)
        thinking_end_token = config_data.get("synthetic_generation_model_think_end_string", None)
        return get_vllm_responses(llm, sampling_params, system_prompts, prompts, thinking_start_token, thinking_end_token)

    if llm_provider == "together":
        together_client = get_together_client()
        together_num_proc = config_data.get("together_num_proc", 1)
        model_name = config_data["synthetic_generation_model_name"]
        temperature = config_data["synthetic_generation_temperature"]
        max_tokens = config_data["synthetic_generation_max_model_len"]
        return get_together_responses(together_client, model_name, temperature, together_num_proc, system_prompts, prompts, max_tokens)

def format_batch(model_name, messages, max_tokens, temperature, run_name=""):
    return [
        {"custom_id": f"r-{i}" + run_name, "body": {
            "model": model_name, "messages": m, 
            "max_tokens": max_tokens,
            "temperature": temperature,
            }
        } for i, m in enumerate(messages)
    ]

def start_batches(config_data, together_client, system_prompts, prompts, input_file_name):
    model_name = config_data["synthetic_generation_model_name"]
    max_tokens = config_data["synthetic_generation_max_model_len"]
    temperature = config_data["synthetic_generation_temperature"]
    force_retry = config_data["batch_force_retry"]

    os.makedirs(BATCH_INPUT_FOLDER, exist_ok=True)
    input_file_path = os.path.join(BATCH_INPUT_FOLDER, input_file_name)
    
    os.makedirs(BATCH_ID_FOLDER, exist_ok=True)
    batch_id_path = os.path.join(BATCH_ID_FOLDER, input_file_name)

    if os.path.isfile(batch_id_path) and not force_retry:
        with open(batch_id_path, "r") as f:
            batch_id = json.load(f)["batch_id"]
        print(f"Batch process already started with ID {batch_id}.")
        return batch_id

    messages = get_conversations(system_prompts, prompts)
    messages = format_batch(model_name, messages, max_tokens, temperature)

    with open(input_file_path, "w") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
        
    ## Uploads batch job file
    file_resp = together_client.files.upload(
        file=input_file_path,
        purpose="batch-api"
    )

    file_id = file_resp.id

    batch = together_client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

    with open(batch_id_path, "w") as f:
        json.dump({"batch_id": batch.id}, f)
        
    return batch.id

def get_batch_responses(config_data, system_prompts, prompts, input_file_name, continue_at_batch_submit):
    input_file_name = f"{input_file_name}.jsonl"

    together_client = get_together_client()

    batch_id = start_batches(config_data, together_client, system_prompts, prompts, input_file_name)
    print(f"Batch processing started with batch id {batch_id}. Feel free to stop this process and run it again later to check if processing has finished.")

    batch = together_client.batches.get_batch(batch_id)
    print(f"Batch status is {batch.status}")

    if continue_at_batch_submit:
        return 
    
    while batch.status in ["IN_PROGRESS", "VALIDATING"]:
        new_batch = together_client.batches.get_batch(batch_id)
        if new_batch.status != batch.status:
            print(f"Batch status is {new_batch.status}...")
            batch = new_batch
        time.sleep(30)

    if batch.status != "COMPLETED":
        raise Exception(f"Batch {batch_id} has failed with status {batch.status}.")

    print("Batch has completed")
    output_file_name = os.path.join(BATCH_OUTPUTS_FOLDER, input_file_name)
    together_client.files.retrieve_content(
        id=batch.output_file_id,
        output=output_file_name,
    )

    with open(output_file_name, "r") as f:
        output_data_list = [json.loads(x) for x in f.readlines()]

    response_body_list = [x["response"]["body"] for x in output_data_list]
    responses = [
        x["choices"][0] if bool(
            len(["choices"][0]) > 1
        ) and bool(
            x["choices"][0]["finish_reason"] == "stop"
        ) else None for x in response_body_list
    ]

    responses = [{
        "reasoning": None,
        "response": None,
    } if x is None else {
        "reasoning": x["message"]["reasoning"],
        "response": x["message"]["content"],
    } for x in responses]

    return responses

def filesafe_parse_name(lang_name):
    return re.sub(r'[^a-zA-Z0-9\_]', '', lang_name)
