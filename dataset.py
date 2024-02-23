import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from fewshot import get_few_shot


def prepare_data(
    example: Dict[str, Union[bool, str]],
    tokenizer: PreTrainedTokenizerBase,
    is_encoder_decoder: bool = False,
    max_length: int = 2048,
    train: bool = False,
    prompt_loss_weight: float = 0.05,
    fewshot: bool = False,
) -> BatchEncoding:
    """
    Prepare data for training or inference.

    Args:
        example ('dict'):
            The example to prepare.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        is_encoder_decoder (`bool`, optional):
            Whether the model is an encoder-decoder model. Defaults to `False`.
        max_length (`int`, optional):
            The maximum length of the input. Defaults to `2048`.
        train (`bool`, optional):
            Whether we are training or not. Defaults to `False`.
        prompt_loss_weight (`float`, optional):
            The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
            of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
        fewshot (`bool`, optional):
            Wheter to add fewshot examples to the prompt. Defaults to `False`.

    Returns:
        `BatchEncoding`: `BatchEncoding` with the prepared data.
    """

    if isinstance(example["label"], bool):
        label = 1 if example["label"] else 0
    elif isinstance(example["label"], str):
        label = 1 if example["label"].lower() == "true" else 0
    elif isinstance(example["label"], int):
        label = example["label"]
    else:
        raise ValueError(f"Label {example['label']} is not a valid label.")

    if tokenizer.chat_template is None:
        if not hasattr(prepare_data, "_warning_logged"):
            logging.warning(
                (
                    "Chat template is not set in the tokenizer. We won't use any chat template for the prompt. "
                    "If you are using an instruction-tuned model, this will likely result in worse performance."
                )
            )
            prepare_data._warning_logged = True
    if not fewshot:
        if tokenizer.chat_template is not None:
            prompt = f"Is the following statement True or False? Answer only True or False. {example['sentence'].strip()}"
        else:
            prompt = f"Is the following statement True or False? {example['sentence'].strip()}"

    else:
        if tokenizer.chat_template is not None:
            prompt = (
                "Is the following statement True or False? Answer only True or False.\n"
                "Here are some examples:\n"
                f"{get_few_shot()}\n\n"
                f"{example['sentence'].strip()}"
            )
        else:
            prompt = (
                "Is the following statement True or False?"
                f"{get_few_shot()}\n\n"
                f"{example['sentence'].strip()}"
            )

            if not is_encoder_decoder:
                prompt = f"{prompt} "  # Add a space at the end so the next token to predict is True or False

    if tokenizer.chat_template is not None:
        prompt_w_answer = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "True" if label == 1 else "False",
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_wo_answer = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    else:
        prompt_wo_answer = prompt
        prompt_w_answer = (
            f"{prompt.strip()} True" if label == 1 else f"{prompt.strip()} False"
        )

    if is_encoder_decoder:
        model_inputs = tokenizer(
            text=prompt_wo_answer,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        model_inputs["labels"] = tokenizer(
            text_target="True" if label == 1 else "False",
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )["input_ids"]

        model_inputs["loss_weight_mask"] = np.ones(
            len(model_inputs["labels"]), dtype=np.float32
        )

    else:
        model_inputs = tokenizer(
            text=prompt_w_answer if train else prompt_wo_answer,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        if train:
            model_inputs["labels"] = model_inputs["input_ids"].copy()

            # Find the prompt length
            prompt_wo_answer = tokenizer(
                text=prompt_wo_answer,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )["input_ids"]

            # Remove the last token if it is an eos token
            if prompt_wo_answer[-1] == tokenizer.eos_token_id:
                prompt_wo_answer = prompt_wo_answer[:-1]

            if len(prompt_wo_answer) > len(model_inputs["labels"]):
                raise ValueError(
                    f"Prompt is longer than the input, something went wrong.nPrompt: {prompt_wo_answer}.\nInput:"
                    f" {model_inputs['labels']}.\n"
                    f"Prompt: {tokenizer.decode(prompt_wo_answer)}\nInput: {tokenizer.decode(model_inputs['labels'])}"
                )

            loss_weight_mask = np.ones(len(model_inputs["labels"]), dtype=np.float32)
            len_prompt = len(prompt_wo_answer)
            len_result = len(model_inputs["labels"]) - len_prompt
            prompt_token_weight = (
                len_result * prompt_loss_weight
            )  # 'prompt_loss_weight' percent of the total loss
            try:
                prompt_token_weight = prompt_token_weight * (
                    len_result / (len_result * (1 - prompt_loss_weight))
                )  # Scale so result tokens can have 1.0 weight
                prompt_token_weight = (
                    prompt_token_weight / len_prompt
                )  # Divide by the number of prompt tokens
            except ZeroDivisionError:
                print(
                    "Found division by zero in prompt token weight calculation. You might have an empty prompt, empty"
                    f" result, or both. Example with error: {example}. Setting prompt token weight to 0.0."
                )
                prompt_token_weight = 0.0

            for i in range(len(prompt_wo_answer)):
                loss_weight_mask[i] = prompt_token_weight

            model_inputs["loss_weight_mask"] = loss_weight_mask

        else:
            if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]

    if "token_type_ids" in model_inputs:
        # LLaMa tokenizer adds token type ids, but we don't need them
        model_inputs.pop("token_type_ids")

    return model_inputs


class ThisIsNotADataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split: str,
        is_encoder_decoder: bool = False,
        max_length: int = 2048,
        fewshot: bool = False,
        prompt_loss_weight: float = 0.05,
        pattern: str = None,
        only_affirmative: bool = False,
        only_negative: bool = False,
        only_non_distractor: bool = False,
        only_distractor: bool = False,
    ):
        self.split = split.lower()
        self.dataset = []
        self.jsonl_dataset = []

        dataset = load_dataset("HiTZ/This-is-not-a-dataset", split=self.split)
        if pattern is not None:
            assert pattern in [
                "Synonymy1",
                "Antonymy1",
                "Synonymy2",
                "Antonymy2",
                "Hypernymy",
                "Part",
                "Substance",
                "Member",
                "Agent",
                "Instrument",
                "Result",
            ]
            print(f"We are only loading examples with pattern {pattern}")

        assert not (only_affirmative and only_negative)

        assert not (only_non_distractor and only_distractor)

        if only_affirmative:
            print("We are only loading affirmative examples")
        if only_negative:
            print("We are only loading negative examples")

        if only_non_distractor:
            print("We are only loading non-distractor examples")
        if only_distractor:
            print("We are only loading distractor examples")

        for example in dataset:
            load = True
            if pattern is not None:
                if example["pattern"] != pattern:
                    load = False
            if only_affirmative:
                if example["negation_type"] == "affirmation":
                    load = False
            if only_negative:
                if example["negation_type"] != "affirmation":
                    load = False
            if only_non_distractor:
                if example["isDistractor"]:
                    load = False
            if only_distractor:
                if not example["isDistractor"]:
                    load = False

            if load:
                self.jsonl_dataset.append(example)

                self.dataset.append(
                    prepare_data(
                        example=example,
                        tokenizer=tokenizer,
                        is_encoder_decoder=is_encoder_decoder,
                        max_length=max_length,
                        fewshot=fewshot,
                        train=self.split == "train",
                        prompt_loss_weight=prompt_loss_weight,
                    )
                )

        print(f"Loaded {len(self.dataset)} examples from {split} split")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_jsonl(self):
        return self.jsonl_dataset


@dataclass
class DataCollatorForSeq2Seq:
    """

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


def get_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    is_encoder_decoder: bool = False,
    max_length: int = 512,
    fewshot: bool = False,
    batch_size: int = 1,
    prompt_loss_weight: float = 0.05,
    num_workers: int = min(8, os.cpu_count()),
    pattern: str = None,
    only_affirmative: bool = False,
    only_negative: bool = False,
    only_non_distractor: bool = False,
    only_distractor: bool = False,
) -> DataLoader:
    """
    Get a dataloader for a dataset.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        split ('list'):
            The split to load (train, dev, test, all).
        is_encoder_decoder (`bool`, optional):
            Whether the model is an encoder-decoder model. Defaults to `False`.
        max_length (`int`, optional):
            The maximum length of the input. Defaults to `2048`.
        fewshot (`bool`, optional):
            Wheter to add fewshot examples to the prompt. Defaults to `False`.
        batch_size (`int`, optional):
            The batch size. Defaults to `1`.
        prompt_loss_weight (`float`, optional):
            The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
            of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
        add_bos_token (`bool`, optional):
            Whether to add the beginning of sentence token to the input. Defaults to `False`.
        num_workers (`int`, optional):
            The number of workers to use for the dataloader. Defaults to `0`.
        pattern (`str`, optional):
            The pattern to use for training. Defaults to `None`.
        only_affirmative (`bool`, optional):
            Whether to only load affirmative examples for training. Defaults to `False`.
        only_negative (`bool`, optional):
            Whether to only load negative examples for training. Defaults to `False`.
        only_non_distractor (`bool`, optional):
            Whether to only load non-distractor examples for training. Defaults to `False`.
        only_distractor (`bool`, optional):
            Whether to only load distractor examples for training. Defaults to `False`.


    Returns:
        `DataLoader`: The dataloader.
    """

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,  # tokenizer.pad_token_id,
        # pad_to_multiple_of=8,  # May be faster on some hardware
    )

    dataset = ThisIsNotADataset(
        tokenizer=tokenizer,
        split=split,
        is_encoder_decoder=is_encoder_decoder,
        max_length=max_length,
        fewshot=fewshot,
        prompt_loss_weight=prompt_loss_weight,
        pattern=pattern,
        only_affirmative=only_affirmative,
        only_negative=only_negative,
        only_non_distractor=only_non_distractor,
        only_distractor=only_distractor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == "train",
        collate_fn=data_collator,
        pin_memory=True,
    )
