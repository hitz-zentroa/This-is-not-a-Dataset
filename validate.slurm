#!/bin/bash
#SBATCH --job-name=NegationLM
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=NegationLM.out.txt
#SBATCH --error=NegationLM.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_ENTITY=igarciaf




accelerate launch run.py configs/zero-shot/FlanT5-xxl.yaml
accelerate launch run.py configs/zero-shot/Vicuna13B.yaml
accelerate launch run.py configs/zero-shot/Llama2-7b.yaml
accelerate launch run.py configs/zero-shot/Llama2-70b.yaml

accelerate launch --mixed_precision bf16 run.py configs/train/Vicuna13B.yaml
accelerate launch --mixed_precision bf16 run.py configs/train/Llama2-7b.yaml
accelerate launch --mixed_precision bf16 run.py configs/train/FlanT5-xxl.yaml

