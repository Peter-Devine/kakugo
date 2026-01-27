---
license: apache-2.0
datasets:
- XXX_DATASET
language:
- XXX_LANGUAGE_CODE
base_model:
- ibm-granite/granite-4.0-micro
pipeline_tag: text-generation
tags:
- low-resource-language
- data-distillation
- conversation
- XXX_LANGUAGE_CODE
- XXX_LANGUAGE_NAME

---
# Kakugo 3B XXX_LANGUAGE_NAME

[[Paper]](https://arxiv.org/abs/2601.14051) [[Code]](https://github.com/Peter-Devine/kakugo) [[Dataset]](https://huggingface.co/datasets/XXX_DATASET)

<div align="center">
    <div style="font-size: 80px;font-family: Arial, Helvetica, sans-serif;font-variant: small-caps;color: #000000;font-weight: 700; margin-top:-40px; margin-bottom:-60px; margin-left: -20px" align="center">Kakugo</div>

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64b63f8ad57e02621dc93c8b/hmRaNkmPAV8rakBOhtgZI.png" alt="Globe Image" width="400"/>
  A data distilled model trained specifically for <strong>XXX_LANGUAGE_NAME</strong>.
</div>
</div>

This is **Kakugo 3B XXX_LANGUAGE_NAME**, a small language model (SLM) fine-tuned to interact with the user in **XXX_LANGUAGE_NAME**.

# How to use

To use this model, you can use your preferred LLM inference package. 

This model should work with any package that supports the original base model [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro).

We provide examples for how to run this with Huggingface or vLLM:

<details>

<summary>Huggingface (Recommended for beginners)</summary>

First, make sure `transformers` is installed on your machine.

```bash
pip install transformers
```

Then run the following Python code to generate a response from the LLM.

```python
from transformers import pipeline

generator = pipeline(model="XXX_MODEL_NAME", task="text-generation")

user_input = input("Please enter your input to the model in XXX_LANGUAGE_NAME:")

do_reasoning = False

open_thinking_tag = "<think>"
close_thinking_tag = "</think>"

if do_reasoning:
    sys_msg = f"Before you respond, first think about your response and enclose your thinking process in {open_thinking_tag} and {close_thinking_tag} delimiters."
else:
    sys_msg = "Be concise in your responses."

message = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_input}
    ]

output = generator(
    message,
    do_sample=False,
    repetition_penalty=1.05,
)

model_response = output[0]["generated_text"][-1]["content"]

if do_reasoning:
    model_response = model_response.split(close_thinking_tag)[-1]

print(model_response)
```

N.B. - We recommend using a `repetition_penalty` of 1.05 as sometimes the model can stuck in a loop of generating repetitive text when generating low-resource languages.

You can set `do_reasoning` to be either True or False to turn "thinking mode" on or off, respectively. If the model is used in thinking mode, then it will take longer to generate a response, but may lead to a better generated response. 
This mode is still experimental, so try both using and not using it for your use-case.

</details>

<br/>

<details>

<summary>vLLM (Recommended for performance)</summary>

First, make sure `vllm` is installed on your machine.

```bash
pip install vllm
```

Then run the following Python code to generate a response from the LLM.

```python
from vllm import LLM, SamplingParams
llm = LLM(model="XXX_MODEL_NAME")

user_input = input("Please enter your input to the model in XXX_LANGUAGE_NAME:")

do_reasoning = True

open_thinking_tag = "<think>"
close_thinking_tag = "</think>"

if do_reasoning:
    sys_msg = f"Before you respond, first think about your response and enclose your thinking process in {open_thinking_tag} and {close_thinking_tag} delimiters."
else:
    sys_msg = "Be concise in your responses."

sampling_params = SamplingParams(temperature=0, repetition_penalty=1.05, max_tokens=2048)

messages = [[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_input}
    ]]

output = llm.chat(messages, sampling_params)

model_response = output[0].outputs[0].text

if do_reasoning:
    model_response = model_response.split(close_thinking_tag)[-1]

print(model_response)
```

N.B. - When using `vllm` for inference of multiple inputs, we recommend inputting them all at once. I.e., add more items to the outer list of the `messages` variable in the above script. [More on vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization).

We recommend using a `repetition_penalty` of 1.05 as sometimes the model can stuck in a loop of generating repetitive text when generating low-resource languages.

You can set `do_reasoning` to be either True or False to turn "thinking mode" on or off, respectively. If the model is used in thinking mode, then it will take longer to generate a response, but may lead to a better generated response. 
This mode is still experimental, so try both using and not using it for your use-case.

</details>

<br/>

# Training data

The training data for this model can be found at [XXX_DATASET](https://huggingface.co/datasets/XXX_DATASET).

This data was created by prompting [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) to generate prompts and responses in XXX_LANGUAGE_NAME.
We also translate a set of prompts and responses from the [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) dataset.

More details about exactly how we created our data can be found in [our paper](XXX_ARCHIV_LINK).

# Training

To make this model, we fine-tuned [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro) for 1 epoch on [XXX_DATASET](https://huggingface.co/datasets/XXX_DATASET) using [Llama Factory](https://github.com/hiyouga/LlamaFactory).

<details>

<summary>Full Llama Factory training hyperparameters</summary>

```yaml
### model
model_name_or_path: ibm-granite/granite-4.0-micro
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset_dir: /workspace/train
dataset: XXX_DATASET
template: granite4
cutoff_len: 8000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
packing: true

### Reporting
report_to: wandb
run_name: XXX_DATASET
logging_steps: 1

### output
output_dir: XXX_DATASET
save_strategy: "no"
save_steps: 99999999
plot_loss: true
overwrite_output_dir: true
save_only_model: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

## eval
val_size: 0.02
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 0.2
```

</details>

<br/>

# Credits

This model was trained by [@ptrdvn](https://huggingface.co/ptrdvn)

If you use this model, please cite:

```bibtex
@article{devine2026kakugo,
  title={Kakugo: Distillation of Low-Resource Languages into Small Language Models},
  author={Devine, Peter and Sanni, Mardhiyah and Adilazuarda, Farid and Loizaga, Julieta Gil and Haddow, Barry},
  journal={arXiv preprint arXiv:2601.14051},
  year={2026}
}
```