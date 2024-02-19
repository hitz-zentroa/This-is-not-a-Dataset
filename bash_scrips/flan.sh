#!/bin/bash
#SBATCH --job-name=TINAD_flan
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=TINAD_flan.out.txt
#SBATCH --error=TINAD_flan.err.txt



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
google/flan-t5-xxl
do

accelerate launch run.py \
  --config configs/zero-shot/FlanT5-xxl.yaml --model_name_or_path "$model_name" --output_dir results/zero-shot/"${model_name//\//_}"

accelerate launch run.py \
 --config configs/zero-shot/base_fewshot.yaml --model_name_or_path "$model_name" --output_dir results/fewshot//"${model_name//\//_}"


done
