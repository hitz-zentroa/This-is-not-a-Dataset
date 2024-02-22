#!/bin/bash
#SBATCH --job-name=TINAD_zero
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --output=TINAD_zero.out.txt
#SBATCH --error=TINAD_zero.err.txt



source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH="$PYTHONPATH:$PWD"

for model_name in \
meta-llama/Llama-2-70b-chat-hf \
meta-llama/Llama-2-70b-hf \
meta-llama/Llama-2-13b-chat-hf \
meta-llama/Llama-2-13b-hf \
meta-llama/Llama-2-7b-chat-hf \
meta-llama/Llama-2-7b-hf \
mistralai/Mistral-7B-Instruct-v0.2 \
HuggingFaceH4/zephyr-7b-beta \
mistralai/Mixtral-8x7B-v0.1 \
mistralai/Mixtral-8x7B-Instruct-v0.1 \
cognitivecomputations/dolphin-2.5-mixtral-8x7b \
NousResearch/Nous-Hermes-2-Yi-34B \
NousResearch/Nous-Hermes-2-SOLAR-10.7B \
NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO \
NousResearch/Nous-Hermes-2-Llama-2-70B \
01-ai/Yi-34B-Chat \
01-ai/Yi-34B \
deepseek-ai/deepseek-llm-7b-chat \
deepseek-ai/deepseek-llm-67b-chat \
Qwen/Qwen1.5-72B-Chat \
Qwen/Qwen1.5-7B-Chat \
google/gemma-2b \
google/gemma-2b-it \
google/gemma-7b \
google/gemma-7b-it \
CohereForAI/aya-101 \
allenai/OLMo-7B \
microsoft/phi-2 \
openchat/openchat-3.5-0106 \
WizardLM/WizardLM-30B-V1.0 \
abacusai/Smaug-72B-v0.1 \
152334H/miqu-1-70b-sf \
alpindale/goliath-120b \
HiTZ/latxa-70b-v1.1
do

# Run the model with 4 bit quantization using data parallel and 4 GPUs (One copy of the model per GPU)
#accelerate launch --multi_gpu --num_processes 4 --main_process_port 29503 run.py \
#  --config configs/zero-shot/base.yaml --model_name_or_path "$model_name" --output_dir results/zero-shot/"${model_name//\//_}"

# Run the model in bfloat16 with deepspeed zero stage 3 using 4 GPUs (Split the model across 4 GPUs)
accelerate launch --use_deepspeed --deepspeed_config_file configs/deepspeed_configs/deepspeed_zero3.json --main_process_port 29503 run.py \
  --config configs/zero-shot/base_deepspeed.yaml --model_name_or_path "$model_name" --output_dir results/zero-shot/"${model_name//\//_}"

done
