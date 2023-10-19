
<p align="center">
    <br>
    <img src="assets/tittle.png" style="height: 250px;">


<p align="center">
    <a href="https://twitter.com/intent/tweet?text=Wow+this+new+model+is+amazing:&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FThis-is-not-a-Dataset"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fhitz-zentroa%2FThis-is-not-a-Dataset"></a>
    <a href="https://github.com/hitz-zentroa/This-is-not-a-Dataset/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/This-is-not-a-Dataset"></a>
    <a href="https://huggingface.co/datasets/HiTZ/This-is-not-a-dataset"><img alt="Public Dataset" src="https://img.shields.io/badge/🤗HuggingFace-Dataset-green"></a>
    <a href="HiTZ/This-is-not-a-dataset"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
<a href="https://www.ehu.eus/en/web/lorea/web-gunea"><img src="https://img.shields.io/badge/LoRea-%20Logic%20and%20Reasoning%20Group-ff3"></a>
    <br>
     <br>
</p>

<h3 align="center">"A Large Negation Benchmark to Challenge Large Language Models"</h3>
    
<p align="justify">
We introduce a large semi-automatically generated dataset of ~400,000 descriptive sentences about commonsense knowledge that can be true or false in which negation is present in about 2/3 of the corpus in different forms that we use to evaluate LLMs.
</p>

- 📖 Paper: [This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models (EMNLP'23)]()
- Dataset available in the 🤗HuggingFace Hub: [HiTZ/This-is-not-a-dataset](https://huggingface.co/datasets/HiTZ/This-is-not-a-dataset)

<p align="justify">
We also provide the code to <b>train</b> and <b>evaluate</b> any LLM in the dataset, as well as the <b>scorer</b> to reproduce the results of the paper.
</p>

## Dataset

The easiest way to download the dataset is using the 🤗HuggingFace Hub. See the [Dataset Card](https://huggingface.co/datasets/HiTZ/This-is-not-a-dataset) for more information about the dataset.

```python
from datasets import load_dataset

dataset = load_dataset("HiTZ/This-is-not-a-dataset")
```

However, we also distribute the dataset in this repository. See [data/README.md](data/README.md) for more information.

## Requirements
The scorer `evaluate.py` does not require any dependency. If you want to run the training or evaluation code you need:
```bash
# Required dependencies
Pytorch>=1.9 (2.1.0 Recommeneded) 
https://pytorch.org/get-started/locally/

transformers
pip install transformers

accelerate 
pip install accelerate

FastChat
pip install fschat

wandb
pip install wandb

# Optional dependencies

bitsandbytes >= 0.40.0 # For 4 / 8 bit quantization
pip install bitsandbytes

PEFT >= 0.4.0 # For LoRA
pip install peft

# You can install all the dependencies with:
pip3 install --upgrade torch transformers accelerate fschat wandb bitsandbytes peft 
```

## Evaluating a LLM

We provide a script to evaluate any LLM in the dataset. First, you need to create a configuration file. 
See [config/zero-shot](config/zero-shot) for an example. This script will evaluate the model in our dataset in zero-shot setting.
Here is an example config to evaluate LLama2-7b Chat:

```yaml
#Model args
model_name_or_path: meta-llama/Llama-2-7b-chat-hf
# Dtype in which we will load the model. You can use bfloat16 is you want to save memory
torch_dtype: "auto"
# Performs quatization using bitsandbytes integration. Allows evaluating LLMs in consumer hardware
quantization: 4
# If force_auto_device_map is set to True. We will split the model into all the available GPUs and CPU, this is useful for large models that do not fit in a single GPU VRAM. 
force_auto_device_map: false
# If set to false, we will sample the probability of generating the True or False tokens (recommended). If set to true, the model will generate a text and we will attempt to locate the string "true" or "false" in the output.
predict_with_generate: false
# Batch size for evaluation. We use auto_batch_finder, so this value is only used to set the maximum batch size, if the batch does not fit in memory, it will be reduced.
per_device_eval_batch_size: 32
# FastChat conversation template to use. See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
conversation_template: llama-2

# dataset arguments
do_train: false
do_eval: false
do_predict: true
# For zero-shot settings, you can evaluate the model in the concatenation of the train, dev and test sets. Set to false to only evaluate in the test set.
do_predict_full_dataset: true
max_seq_length: 512

# Output Dir
output_dir: results/zero-shot/llama-2-7b-chat-hf
````

Once you have created the config file, you can run the evaluation script:

```bash
accelerate launch run.py configs/zero-shot/Llama2-7b.yaml
```

You can use accelerate to run the evaluation in multiple GPUs. See [accelerate documentation](https://github.com/huggingface/accelerate) for more information.
```bash
accelerate launch --multi_gpu --num_processes 2 run.py configs/zero-shot/Llama2-7b.yaml
```

## Training a LLM
You can train a LLMs in our dataset. First, you need to create a configuration file. See [config/finetune](config/finetune) for an example. Here is an example config to finetune LLama2-7b Chat:

```yaml
#Model args
model_name_or_path: meta-llama/Llama-2-7b-chat-hf
torch_dtype: "float32"
# We use LoRA for efficient training. Without LoRA you would need 4xA100 to train Llama2-7b Chat. See https://arxiv.org/abs/2106.09685
use_lora: true
quantization: 4
predict_with_generate: false
conversation_template: llama-2
force_auto_device_map: false

# Dataset arguments
do_train: true
do_eval: true
do_predict: true
do_predict_full_dataset: false
max_seq_length: 512

# Train only on a pattern i.e Synonymy1, Hypernymy, etc...
pattern: null
# Train only on affirmative sentences
only_affirmative: False
# Train only on negated sentences
only_negated: False
# Train only on sentences without a distractor
only_non_distractor: False
# Train only on sentences with a distractor
only_distractor: False

#Training arguments
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
per_device_eval_batch_size: 32
optim: paged_adamw_32bit
learning_rate: 0.0003
weight_decay: 0
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03

# Output Dir
output_dir: results/finetune/Llama-2-7b-chat-hf
```

Once you have created the config file, you can run the training script:

```bash
# Single GPU with bfloat16 mixed precision
accelerate launch --mixed_precision bf16 run.py configs/train/Llama2-7b.yaml
# Multi GPU with bfloat16 mixed precision
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 run.py configs/train/Llama2-7b.yaml
```



## Scorer
If you use our `run.py` script, the models will be automatically evaluated. But you might want to evaluate results generated by your custom code. 
In that case, you can use the `evaluate.py` script. The scorer 
expects a `.jsonl` file as input similar to the dataset files, but with the extra field `prediction`. This field 
should contain the prediction for each example as a boolean `true` or `false`. Each line should be a dictionary. For example:
```jsonlines
{"pattern_id": 1, "pattern": "Synonymy1", "test_id": 0, "negation_type": "affirmation", "semantic_type": "none", "syntactic_scope": "none", "isDistractor": false, "label": true, "sentence": "An introduction is commonly the first section of a communication.", "prediction": true}
{"pattern_id": 1, "pattern": "Synonymy1", "test_id": 0, "negation_type": "affirmation", "semantic_type": "none", "syntactic_scope": "none", "isDistractor": true, "label": false, "sentence": "An introduction is commonly the largest possible quantity.", "prediction": false}
...
```

### Result interpretation
The scorer will output the following metrics:
- **all_affirmations**: Accuracy of the model in affirmative sentences
- **all_negations**: Accuracy of the model in negated sentences
- **all**: (Overall) Accuracy of the model in all sentences
- **input_affirmation**: Accuracy of the model in affirmative sentences without distractors
- **input_negation**: Accuracy of the model in negated sentences without distractors
- **distractor_affirmation**: Accuracy of the model in affirmative sentences with distractors
- **distractor_negation**: Accuracy of the model in negated sentences with distractors
- **Negation_analysis**: Fine-grained analysis of the model in negated sentences (verbal, analytic, clausal, non_verbal, synthetic, subclausal negation types)
- **Synonymy1, Hypernymy, Part...**: Fine-grained analysis of the model in each pattern

You can call the scorer with the following command:

```bash
python3 evaluate.py --predictions_path <path_to_input_file>.jsonl --output_path <path_to_output_scores>.json
```


# Citation
The paper will be presented at EMNLP 2023, the citation will be available soon. For now, you can use the following bibtex:

```bibtex
@inproceedings{this-is-not-a-dataset,
    title = "This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models",
    author = "Iker García-Ferrero, Begoña Altuna, Javier Alvez, Itziar Gonzalez-Dios, German Rigau",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```