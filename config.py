from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The local path or huggingface hub name of the model and tokenizer to use."
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this"
                " dtype. If `auto` is passed, the dtype will be automatically derived"
                " from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use LoRA. If True, the model will be trained with LoRA: https://arxiv.org/abs/2106.09685"
            )
        },
    )

    quantization: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes"
            )
        },
    )
    lora_weights_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If the model has been trained with LoRA, "
                "path or huggingface hub name or local path to the pretrained weights."
            )
        },
    )

    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "Lora attention dimension."},
    )

    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."},
    )

    lora_target_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": (
                "The target modules to which LoRA will be applied. If not specified, We"
                " will use the default modules for the model in huggingface PEFT library."
            )
        },
    )

    conversation_template: str = field(
        default=None,
        metadata={
            "help": (
                "The config template to use to generate conversations. See "
                "https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py for more details"
            )
        },
    )

    add_bos_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add the BOS token to the beginning of the prompt (Encoder-only models). Defaults to False."
            )
        },
    )

    use_flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the FlashAttention. If True, we will use FlashAttention. Be careful, not all models "
                "support FlashAttention. See https://github.com/huggingface/transformers/issues/26350. "
                "Defaults to False."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    do_predict_full_dataset: bool = field(
        default=False,
        metadata={
            "help": "Whether to run predictions on the full dataset. If True, the model will be evaluated on the "
            "full dataset. If False, the model will be evaluated on the test set. Defaults to False."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences"
                " longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    prompt_loss_weight: float = field(
        default=0.05,
        metadata={
            "help": (
                "The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total"
                " weight of 5% in the loss while the result tokens will have a total weight of 95%. Only used for"
                " computing the loss in the training data. Defaults to `0.05`."
            )
        },
    )

    force_auto_device_map: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to force the use of the auto device map. If set to True, the model will be split across "
                "GPUs and CPU to fit the model in memory. If set to False, a full copy of the model will be loaded "
                "into each GPU. Defaults to False."
            )
        },
    )

    pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The pattern to use for training. If not specified, all patterns will be used."
            ),
            "choices": [
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
            ],
        },
    )

    only_affirmative: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only load affirmative examples for training. Defaults to `False`."
            )
        },
    )

    only_negative: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only load negative examples for training. Defaults to `False`."
            )
        },
    )

    only_non_distractor: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only load non-distractor examples for training. Defaults to `False`."
            )
        },
    )

    only_distractor: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only load distractor examples for training. Defaults to `False`."
            )
        },
    )
