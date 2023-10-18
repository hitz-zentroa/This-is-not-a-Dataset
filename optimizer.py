from transformers import Seq2SeqTrainingArguments, PreTrainedModel
import torch
from transformers.utils import ExplicitEnum
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import torch.nn as nn


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"


def get_optimizer(training_args: Seq2SeqTrainingArguments, model: PreTrainedModel):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {"lr": training_args.learning_rate}

    adam_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    if training_args.optim == OptimizerNames.ADAFACTOR:
        from transformers.optimization import Adafactor

        optimizer_cls = Adafactor
        optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
    elif training_args.optim == OptimizerNames.ADAMW_HF:
        from transformers.optimization import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif training_args.optim in [
        OptimizerNames.ADAMW_TORCH,
        OptimizerNames.ADAMW_TORCH_FUSED,
    ]:
        from torch.optim import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
        if training_args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
            optimizer_kwargs.update({"fused": True})
    elif training_args.optim == OptimizerNames.ADAMW_TORCH_XLA:
        try:
            from torch_xla.amp.syncfree import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
    elif training_args.optim == OptimizerNames.ADAMW_APEX_FUSED:
        try:
            from apex.optimizers import FusedAdam

            optimizer_cls = FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError(
                "Trainer tried to instantiate apex FusedAdam but apex is not installed!"
            )
    elif training_args.optim in [
        OptimizerNames.ADAMW_BNB,
        OptimizerNames.ADAMW_8BIT,
        OptimizerNames.PAGED_ADAMW,
        OptimizerNames.PAGED_ADAMW_8BIT,
        OptimizerNames.LION,
        OptimizerNames.LION_8BIT,
        OptimizerNames.PAGED_LION,
        OptimizerNames.PAGED_LION_8BIT,
    ]:
        try:
            from bitsandbytes.optim import AdamW, Lion

            is_paged = False
            optim_bits = 32
            optimizer_cls = None
            additional_optim_kwargs = adam_kwargs
            if "paged" in training_args.optim:
                is_paged = True
            if "8bit" in training_args.optim:
                optim_bits = 8
            if "adam" in training_args.optim:
                optimizer_cls = AdamW
            elif "lion" in training_args.optim:
                optimizer_cls = Lion
                additional_optim_kwargs = {
                    "betas": (training_args.adam_beta1, training_args.adam_beta2)
                }

            bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
            optimizer_kwargs.update(additional_optim_kwargs)
            optimizer_kwargs.update(bnb_kwargs)
        except ImportError:
            raise ValueError(
                "Trainer tried to instantiate bnb optimizer but bnb is not installed!"
            )
    elif training_args.optim == OptimizerNames.ADAMW_BNB:
        try:
            from bitsandbytes.optim import Adam8bit

            optimizer_cls = Adam8bit
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError(
                "Trainer tried to instantiate bnb Adam8bit but bnb is not installed!"
            )
    elif training_args.optim == OptimizerNames.ADAMW_ANYPRECISION:
        raise NotImplementedError("AdamWAnyprecision is not supported")
    elif training_args.optim == OptimizerNames.SGD:
        optimizer_cls = torch.optim.SGD
    elif training_args.optim == OptimizerNames.ADAGRAD:
        optimizer_cls = torch.optim.Adagrad
    else:
        raise ValueError(
            f"Trainer cannot instantiate unsupported optimizer: {training_args.optim}"
        )

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                print(f"skipped {module}: {skipped / 2 ** 20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                print(f"bitsandbytes: will optimize {module} in fp32")
        print(f"skipped: {skipped / 2 ** 20}M params")

    return optimizer
