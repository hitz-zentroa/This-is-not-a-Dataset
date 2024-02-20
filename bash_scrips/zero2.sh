#!/bin/bash
#SBATCH --job-name=TINAD_zero2
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=TINAD_zero2.out.txt
#SBATCH --error=TINAD_zero2.err.txt



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
allenai/OLMo-7B \
Qwen/Qwen1.5-72B-Chat \
Qwen/Qwen1.5-7B-Chat \
01-ai/Yi-34B-Chat \
01-ai/Yi-34B \
openchat/openchat-3.5-0106 \
NousResearch/Nous-Hermes-2-Yi-34B \
NousResearch/Nous-Hermes-2-SOLAR-10.7B \
NousResearch/Nous-Hermes-2-Llama-2-70B \
WizardLM/WizardLM-30B-V1.0 \
cognitivecomputations/dolphin-2.5-mixtral-8x7b \
deepseek-ai/deepseek-llm-7b-chat \
deepseek-ai/deepseek-llm-67b-chat \
abacusai/Smaug-72B-v0.1 \
152334H/miqu-1-70b-sf \
alpindale/goliath-120b 
do



accelerate launch --multi_gpu --num_processes 2 --main_process_port 29503 run.py \
  --config configs/zero-shot/base.yaml --model_name_or_path "$model_name" --output_dir results/zero-shot/"${model_name//\//_}"

done
