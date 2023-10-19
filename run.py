from load_model import load_model
from dataset import get_dataloader
from evaluate import evaluate
import torch
import os
from config import DataTrainingArguments, ModelArguments
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    get_scheduler,
)

from tqdm import tqdm
from accelerate import Accelerator, find_executable_batch_size
import wandb
from typing import List
import gc
import json
import math
import sys
from optimizer import get_optimizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
import logging


def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""

    print(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")


def compute_loss(model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.
    Subclass and override for custom behavior.
    """

    if "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        raise ValueError("You should supply a labels key to compute the loss")

    if "loss_weight_mask" in inputs:
        loss_weight_mask = inputs.pop("loss_weight_mask")
    else:
        raise ValueError("You should supply a loss_weight_mask key to compute the loss")

    if unwrap_model(model).config.is_encoder_decoder:
        outputs = model(labels=labels, **inputs)
    else:
        outputs = model(**inputs)

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

    model_name = unwrap_model(model)._get_name()
    if (
        model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
        or model_name == "PeftModelForCausalLM"
    ):
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        loss_weight_mask = loss_weight_mask[..., 1:].contiguous()

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss_weight_mask = loss_weight_mask.view(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    loss = loss_fct(logits, labels)
    loss = torch.sum(loss * loss_weight_mask) / torch.sum(loss_weight_mask)

    return (loss, outputs) if return_outputs else loss


def gen_predictions(
    model,
    tokenizer,
    true_tokens_ids: List[int],
    false_tokens_ids: List[int],
    dataloader,
    output_path,
    accelerator,
    print_first=False,
    predict_with_generate=False,
    return_scores=False,
):
    if predict_with_generate and return_scores:
        raise ValueError(
            "return_scores is not supported when predict_with_generate is True"
        )
    model.eval()
    with torch.no_grad():
        samples_seen: int = 0
        yes_id = true_tokens_ids[0]
        no_id = false_tokens_ids[0]
        all_preds = []
        all_scores = []
        first = True

        for step, batch in enumerate(
            tqdm(dataloader, f"Inference on {os.path.basename(output_path)}")
        ):
            if print_first and accelerator.is_local_main_process:
                ### DEBUG ###
                if print_first and first and accelerator.is_main_process:
                    decodeable_inputs = batch.input_ids.clone()
                    decodeable_inputs[
                        decodeable_inputs == -100
                    ] = tokenizer.pad_token_id

                    model_inputs = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_inputs,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    print(f"*** Sample of batch 0 ***")
                    print(f"-- Model inputs --\n{model_inputs}")
                    print(f"*** End of sample ***\n")
                    first = False

            if not predict_with_generate:
                if not model.config.is_encoder_decoder:
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                else:
                    encoder_output = model.get_encoder()(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    decoder_args = {
                        "attention_mask": batch["attention_mask"],
                        "use_cache": False,
                        "encoder_outputs": encoder_output,
                    }

                    gen_inputs = model.prepare_inputs_for_generation(
                        input_ids=torch.tensor(
                            [[tokenizer.pad_token_id]] * len(batch["input_ids"])
                        ).to(batch["input_ids"].device),
                        **decoder_args,
                    )

                    logits = model(
                        **gen_inputs,
                    ).logits

                logits = logits[:, -1, :]
                logits = torch.nn.functional.softmax(logits, dim=-1)
                logits = logits[:, [yes_id, no_id]]
                logits = logits[:, 0] / (logits[:, 0] + logits[:, 1])
                preds = logits > 0.5
                preds = accelerator.gather(preds).cpu().tolist()
                logits = accelerator.gather(logits).cpu().tolist()

                if accelerator.is_local_main_process:
                    if accelerator.num_processes > 1:
                        # Remove duplicated in last batch if we are in a distributed setting
                        if step == len(dataloader) - 1:
                            preds = preds[: (len(dataloader.dataset) - samples_seen)]
                            logits = logits[: (len(dataloader.dataset) - samples_seen)]
                        else:
                            samples_seen += len(batch)

                    all_preds.extend(preds)
                    all_scores.extend(logits)
            else:
                preds = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=6,
                )
                preds = accelerator.gather(
                    accelerator.pad_across_processes(
                        preds,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                ).cpu()

                inputs_ids = accelerator.gather(
                    accelerator.pad_across_processes(
                        batch["input_ids"],
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                ).cpu()

                preds = preds[:, len(inputs_ids[0]) :]

                if accelerator.is_local_main_process:
                    if accelerator.num_processes > 1:
                        # Remove duplicated in last batch if we are in a distributed setting
                        if step == len(dataloader) - 1:
                            preds = preds[: (len(dataloader.dataset) - samples_seen)]
                        else:
                            samples_seen += len(batch)

                    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                    # print(preds)
                    for pred in preds:
                        pred = pred.lower()

                        if "true" in pred:
                            all_preds.append(True)
                        else:
                            all_preds.append(False)

        if accelerator.is_local_main_process:
            with open(output_path, "w", encoding="utf8") as f:
                for pred in all_preds if not return_scores else all_scores:
                    print(pred, file=f)

            if not return_scores:
                json_dataset = dataloader.dataset.get_jsonl()
                assert len(json_dataset) == len(all_preds)
                with open(
                    os.path.splitext(output_path)[0] + ".jsonl", "w", encoding="utf8"
                ) as f:
                    for json_line, pred in zip(json_dataset, all_preds):
                        json_line["prediction"] = bool(pred)
                        print(json.dumps(json_line, ensure_ascii=False), file=f)

    model.train()


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
):
    assert (
        training_args.do_train or training_args.do_predict
    ), "You must specify do_train or do_predict"
    assert not (training_args.do_train and data_args.do_predict_full_dataset), (
        "You cannot do both training and predict_full_dataset, "
        "as the model will be evaluated on the full dataset, which"
        " includes the training set."
    )

    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator()
    print(f"Accelerator State: {accelerator.state}")

    set_seed(training_args.seed)

    if training_args.do_train:
        model, tokenizer = load_model(
            inference=False,
            model_weights_name_or_path=model_args.model_name_or_path,
            lora_weights_name_or_path=model_args.lora_weights_name_or_path,
            quantization=model_args.quantization,
            use_lora=model_args.use_lora,
            lora_target_modules=model_args.lora_target_modules,
            torch_dtype=model_args.torch_dtype,
            force_auto_device_map=data_args.force_auto_device_map,
            use_flash_attention=model_args.use_flash_attention,
            use_gradient_checkpointing=model_args.use_lora,
        )

        true_tokens_ids = tokenizer.encode("True", add_special_tokens=False)
        false_tokens_ids = tokenizer.encode("False", add_special_tokens=False)

        train_dataloader = get_dataloader(
            tokenizer=tokenizer,
            split="train",
            is_encoder_decoder=model.config.is_encoder_decoder,
            max_length=data_args.max_seq_length,
            conv_template=model_args.conversation_template,
            batch_size=training_args.per_device_train_batch_size,
            prompt_loss_weight=data_args.prompt_loss_weight,
            add_bos_token=model_args.add_bos_token,
            pattern=data_args.pattern,
            only_negated=data_args.only_negated,
            only_affirmative=data_args.only_affirmative,
            only_distractor=data_args.only_non_distractor,
            only_non_distractor=data_args.only_non_distractor,
        )

        dev_dataloader = None
        if training_args.do_eval:
            dev_dataloader = get_dataloader(
                tokenizer=tokenizer,
                split="validation",
                is_encoder_decoder=model.config.is_encoder_decoder,
                max_length=data_args.max_seq_length,
                conv_template=model_args.conversation_template,
                batch_size=training_args.per_device_train_batch_size,
                prompt_loss_weight=data_args.prompt_loss_weight,
                add_bos_token=model_args.add_bos_token,
                pattern=data_args.pattern,
                only_negated=data_args.only_negated,
                only_affirmative=data_args.only_affirmative,
                only_distractor=data_args.only_non_distractor,
                only_non_distractor=data_args.only_non_distractor,
            )

        if accelerator.is_main_process:
            wandb.init(
                project="ThisIsNotADataset",
                name=f"{os.path.basename(training_args.output_dir)}",
                config=vars(training_args),
            )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / training_args.gradient_accumulation_steps
        )
        max_train_steps = int(
            training_args.num_train_epochs * num_update_steps_per_epoch
        )

        optimizer = get_optimizer(training_args=training_args, model=model)

        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * max_train_steps),
            num_training_steps=max_train_steps,
        )

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

        if dev_dataloader is not None:
            dev_dataloader = accelerator.prepare(dev_dataloader)

        completed_steps = 0
        best_epoch_metric: float = -1
        validation_dir: str = os.path.join(training_args.output_dir, "val_logs")
        os.makedirs(validation_dir, exist_ok=True)
        running_loss = 0
        num_batches = 0
        first = True

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ascii=True,
            desc="Training",
        )

        for epoch in range(int(training_args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_dataloader):
                ### DEBUG ###
                if first and accelerator.is_main_process:
                    decodeable_inputs = batch.input_ids.clone()
                    decodeable_inputs[
                        decodeable_inputs == -100
                    ] = tokenizer.pad_token_id

                    model_inputs = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_inputs,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    decodeable_labels = batch.labels.clone()
                    decodeable_labels[
                        decodeable_labels == -100
                    ] = tokenizer.pad_token_id

                    labels = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_labels,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    print(f"*** Sample of batch 0 ***")
                    print(f"-- Model inputs --\n{model_inputs}")
                    print(f"-- Labels --\n{labels}")
                    print(f"*** End of sample ***\n")
                    first = False

                loss = compute_loss(model=model, inputs=batch, return_outputs=False)

                running_loss += loss.item()
                loss = loss / training_args.gradient_accumulation_steps
                accelerator.backward(loss)
                num_batches += 1

                if (
                    step % training_args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if (
                        accelerator.is_local_main_process
                        and completed_steps > 0
                        and (completed_steps % 10 == 0)
                    ):
                        wandb.log(
                            {
                                "Train/Loss": loss.item(),
                                "Train/Running Loss": loss.item() / num_batches,
                                "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": completed_steps,
                            }
                        )

                    if (
                        training_args.eval_steps is not None
                        and completed_steps % training_args.eval_steps == 0
                        and dev_dataloader is not None
                    ):
                        gen_predictions(
                            model=model,
                            tokenizer=tokenizer,
                            true_tokens_ids=true_tokens_ids,
                            false_tokens_ids=false_tokens_ids,
                            dataloader=dev_dataloader,
                            output_path=os.path.join(
                                validation_dir,
                                f"step_{completed_steps}.preds",
                            ),
                            accelerator=accelerator,
                            predict_with_generate=training_args.predict_with_generate,
                        )

                        if accelerator.is_main_process:
                            results = evaluate(
                                predictions_path=os.path.join(
                                    validation_dir,
                                    f"step_{completed_steps}.jsonl",
                                ),
                                output_path=os.path.join(
                                    validation_dir,
                                    f"step_{completed_steps}_results.json",
                                ),
                            )

                            results["step"] = completed_steps

                            wandb.log(results)

                            accuracy = results["all"]["accuracy"]

                            if (
                                (accuracy >= best_epoch_metric)
                                or (best_epoch_metric < 0)
                                or (math.isnan(best_epoch_metric))
                            ):
                                print(
                                    f"New best model :) step {completed_steps} "
                                    f"PrevF1 {best_epoch_metric} accuracy {accuracy}"
                                )
                                best_epoch_metric = accuracy
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(
                                    training_args.output_dir,
                                    save_function=accelerator.save,
                                )
                                tokenizer.save_pretrained(training_args.output_dir)
                            else:
                                print(
                                    f"This epoch did not improve :( Step {completed_steps} "
                                    f"PrevF1 {best_epoch_metric} accuracy {accuracy}"
                                )

                        accelerator.wait_for_everyone()
                        model.train()

            if (epoch > training_args.eval_delay) and dev_dataloader is not None:
                gen_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    true_tokens_ids=true_tokens_ids,
                    false_tokens_ids=false_tokens_ids,
                    dataloader=dev_dataloader,
                    output_path=os.path.join(
                        validation_dir,
                        f"step_{completed_steps}.preds",
                    ),
                    accelerator=accelerator,
                    predict_with_generate=training_args.predict_with_generate,
                )

                if accelerator.is_main_process:
                    results = evaluate(
                        predictions_path=os.path.join(
                            validation_dir,
                            f"step_{completed_steps}.jsonl",
                        ),
                        output_path=os.path.join(
                            validation_dir,
                            f"step_{completed_steps}_results.json",
                        ),
                    )

                    results["step"] = completed_steps

                    wandb.log(results)

                    accuracy = results["all"]["accuracy"]

                    if (
                        (accuracy >= best_epoch_metric)
                        or (best_epoch_metric < 0)
                        or (math.isnan(best_epoch_metric))
                    ):
                        print(
                            f"New best model :) step {completed_steps} "
                            f"PrevF1 {best_epoch_metric} accuracy {accuracy}"
                        )
                        best_epoch_metric = accuracy
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            training_args.output_dir,
                            save_function=accelerator.save,
                        )
                        tokenizer.save_pretrained(training_args.output_dir)
                    else:
                        print(
                            f"This epoch did not improve :( Step {completed_steps} "
                            f"PrevF1 {best_epoch_metric} accuracy {accuracy}"
                        )

                accelerator.wait_for_everyone()
                model.train()

        progress_bar.close()

        if accelerator.is_main_process:
            wandb.finish()

    clean_cache()

    if training_args.do_predict:
        if training_args.do_train:
            print(
                "You are doing inference after training a model! We will load the "
                f"pretrained model saved in {training_args.output_dir}."
            )
            if model_args.use_lora:
                lora_weights_name_or_path = training_args.output_dir
                model_path = model_args.model_name_or_path

            else:
                model_path = training_args.output_dir
                lora_weights_name_or_path = None

        else:
            model_path = model_args.model_name_or_path
            lora_weights_name_or_path = model_args.lora_weights_name_or_path

        model, tokenizer = load_model(
            inference=True,
            model_weights_name_or_path=model_path,
            use_lora=lora_weights_name_or_path is not None,
            quantization=model_args.quantization,
            lora_weights_name_or_path=lora_weights_name_or_path,
            force_auto_device_map=data_args.force_auto_device_map,
            use_flash_attention=model_args.use_flash_attention,
        )

        true_tokens_ids = tokenizer.encode("True", add_special_tokens=False)
        false_tokens_ids = tokenizer.encode("False", add_special_tokens=False)

        # model = accelerator.prepare(model)

        first = True

        @find_executable_batch_size(
            starting_batch_size=training_args.per_device_eval_batch_size
        )
        def inference(batch_size):
            nonlocal model, tokenizer, data_args, model_args, training_args, first, true_tokens_ids, false_tokens_ids

            print(f"Inference with batch size {batch_size}")
            test_dataloader = get_dataloader(
                tokenizer=tokenizer,
                split="test" if not data_args.do_predict_full_dataset else "all",
                is_encoder_decoder=model.config.is_encoder_decoder,
                max_length=data_args.max_seq_length,
                conv_template=model_args.conversation_template,
                batch_size=batch_size,
                add_bos_token=model_args.add_bos_token,
            )

            model, test_dataloader = accelerator.prepare(model, test_dataloader)

            gen_predictions(
                model=model,
                tokenizer=tokenizer,
                true_tokens_ids=true_tokens_ids,
                false_tokens_ids=false_tokens_ids,
                dataloader=test_dataloader,
                output_path=os.path.join(training_args.output_dir, f"test.preds"),
                accelerator=accelerator,
                print_first=first,
                predict_with_generate=training_args.predict_with_generate,
            )
            first = False

        inference()

        if accelerator.is_main_process:
            evaluate(
                predictions_path=os.path.join(training_args.output_dir, f"test.jsonl"),
                output_path=os.path.join(
                    training_args.output_dir, f"test_results.json"
                ),
            )

    clean_cache()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    print(f"Sys args {sys.argv}")

    if len(sys.argv) > 0 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        print(f"Loading json config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1])
        )

    elif len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
        # If we pass only one argument to the script, and it's the path to a yaml file,
        # let's parse it to get our arguments.
        print(f"Loading yaml config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[-1])
        )
    else:
        print("No config file passed, using command line arguments.")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    main(model_args, data_args, training_args)
