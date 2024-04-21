#!/bin/bash
#SBATCH --job-name=TINAD_fewshot
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --output=TINAD_fewshot.out.txt
#SBATCH --error=TINAD_fewshot.err.txt



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
meta-llama/Meta-Llama-3-70B-Instruct \
CohereForAI/c4ai-command-r-v01 \
CohereForAI/c4ai-command-r-plus \
mistralai/Mixtral-8x22B-Instruct-v0.1 \
mistralai/Mixtral-8x22B-v0.1  \
alpindale/WizardLM-2-8x22B  \
lucyknada/microsoft_WizardLM-2-7B
do


# Run the model with 4 bit quantization using data parallel and 4 GPUs (One copy of the model per GPU)
accelerate launch --multi_gpu --num_processes 4 --main_process_port 29507 run.py \
 --config configs/zero-shot/base_fewshot.yaml --model_name_or_path "$model_name" --output_dir results/fewshot//"${model_name//\//_}"

# Run the model in bfloat16 with deepspeed zero stage 3 using 4 GPUs (Split the model across 4 GPUs)
#accelerate launch --use_deepspeed --num_processes 4 --deepspeed_config_file configs/deepspeed_configs/deepspeed_zero3.json --main_process_port 29507 run.py \
# --config configs/zero-shot/base_fewshot_deepspeed.yaml --model_name_or_path "$model_name" --output_dir results/fewshot//"${model_name//\//_}"
done
