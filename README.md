<div align="center">

  # Kakugo

  <img width="300" alt="Globe Image" src="https://github.com/user-attachments/assets/451166a2-af70-4b73-9b37-4c019d9a54b1" />

  <em>Distil low resource language knowledge from large to small language models</em>

</div>

[[Paper]](https://arxiv.org/abs/2601.14051)    |   [[Models]](https://hf.co/collections/ptrdvn/kakugo-models)      |  [[Datasets]](https://hf.co/collections/ptrdvn/kakugo-datasets)

## How to generate data

To run this code, you need either local GPU(s) that are capable of running the teacher model (GPT-OSS 120B by default) or a [Together AI](https://www.together.ai/) account.

If you will then need the following dependencies:

```bash
pip install pandas datasets transformers numpy
```

And then to use Together AI:

```bash
pip install together
```

Or to use your local GPU:

```bash
pip install vllm
```

Before running, check the settings in `./config.yaml` to make sure they are to your liking, or leave them as the default.

Then, simply set the language name like so:

```bash
export LANG_NAME="Javanese"
```

To run the data preparation process, you need to run either:

**For local GPUs**

```bash
python run_prompt_gen.py --language "$LANG_NAME" --llm_provider vllm
python run_response_gen.py --language "$LANG_NAME" --llm_provider vllm 
```

**For Together AI**

```bash
python run_prompt_gen.py --language "$LANG_NAME" --llm_provider together
python run_response_gen.py --language "$LANG_NAME" --llm_provider together 
```

**For Together AI with batch processing**

If you would like to use batch processing so that your process does not hang, you can set `do_batch: True` in `./config.yaml` and then run the following 

```bash
export TOGETHER_AI_KEY=YOUR_API_KEY

python run_prompt_gen.py --language "$LANG_NAME" --llm_provider together # This will not run as a batch as it does not take very long usually
python run_response_gen.py --language "$LANG_NAME" --llm_provider together --continue_at_batch_submit
```

Then wait a while (usually a few hours, but can be up to 24 hours) and then run this command again:

```bash
export TOGETHER_AI_KEY=YOUR_API_KEY

python run_response_gen.py --language "$LANG_NAME" --llm_provider together
```

Where it should pick up all the finished batches.

## How to train model

First, you must prepare the training data by running:

```bash
python ./run_train_prep.py --language "$LANG_NAME"
```

In our experiments, we trained models by first running a LlamaFactory with a docker command (so you will need to install Docker if you dont currently have it).

Feel free to get LlamaFactory working in any way you see fit by [following their helpful guides](https://github.com/hiyouga/LlamaFactory?tab=readme-ov-file#install-from-docker-image). 

<details>

<summary>Here is how we managed to get it running successfully on our 8 x 3090 system.</summary>

```bash
REPO_DIR=~/projects/kakugo/train
docker run -p 7860:7860 --shm-size 64g -v ~/.netrc:/root/.netrc -v ~/.config:/root/.config/ -v ~/.cache:/root/.cache/ -v $REPO_DIR:/workspace/train/ --gpus '"all"' --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 --device /dev/nvidia2:/dev/nvidia2 --device /dev/nvidia3:/dev/nvidia3 --device /dev/nvidia4:/dev/nvidia4 --device /dev/nvidia5:/dev/nvidia5 --device /dev/nvidia6:/dev/nvidia6 --device /dev/nvidia7:/dev/nvidia7 --device /dev/nvidia-caps/ --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --rm -it --ipc=host hiyouga/llamafactory:latest
```

Where repo dir is where you have this repo saved to. The extra arguments were to prevent GPU errors that were probably specific to our set-up. 

</details>


Once inside the LlamaFactory image, we ran:

```bash
pip install wandb # For WandB logging
export LANG_NAME="Javanese" # !!!NB!!! Here, you need to write the name of your language identically to the data creation step (and case sensitive), but without any non-alphabetic characters.
# This includes removing any spaces, apostrophes, or brackets, for example.
FORCE_TORCHRUN=1 llamafactory-cli train /workspace/train/train_configs/genreas_trans_full_$LANG_NAME.yaml
```

Then, the final model should be able to be found in `./train/train_outputs`, saved in Huggingface format so you should be able to use it with your favourite inference library, such as [vLLM](https://docs.vllm.ai/en/stable/getting_started/quickstart/), for example.
