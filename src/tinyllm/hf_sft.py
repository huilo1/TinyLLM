from __future__ import annotations

import argparse

from datasets import load_from_disk
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from tinyllm.hf_config import HFExperimentConfig, load_hf_config
from tinyllm.utils import ensure_dir, save_json, set_seed


def _compute_dtype(config: HFExperimentConfig) -> torch.dtype:
    if config.model.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _quantization_config(config: HFExperimentConfig, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if not config.model.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_tokenizer(config: HFExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(config: HFExperimentConfig, compute_dtype: torch.dtype):
    quantization_config = _quantization_config(config, compute_dtype)
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        **model_kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return model


def _create_lora_config(config: HFExperimentConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )


def _load_datasets(config: HFExperimentConfig):
    processed_dir = config.data.processed_dir
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dir}. Run tinyllm-hf-prepare first."
        )

    dataset = load_from_disk(str(processed_dir))
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] if "validation" in dataset and len(dataset["validation"]) > 0 else None
    return train_dataset, eval_dataset


def _create_training_args(config: HFExperimentConfig) -> SFTConfig:
    use_bf16 = config.model.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return SFTConfig(
        output_dir=str(config.checkpoints_dir),
        max_length=config.model.max_length,
        packing=config.training.packing,
        assistant_only_loss=config.training.assistant_only_loss,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        logging_first_step=True,
        remove_unused_columns=False,
    )


def train(config: HFExperimentConfig, resume_from_checkpoint: str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HF QLoRA training requires a CUDA GPU.")

    set_seed(config.run.seed)
    ensure_dir(config.run_dir)
    ensure_dir(config.checkpoints_dir)
    ensure_dir(config.adapter_dir)

    compute_dtype = _compute_dtype(config)
    tokenizer = _load_tokenizer(config)
    train_dataset, eval_dataset = _load_datasets(config)
    model = _load_model(config, compute_dtype)
    lora_config = _create_lora_config(config)
    if eval_dataset is None:
        config.training.eval_strategy = "no"
    training_args = _create_training_args(config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    metadata = {
        "run_name": config.run.name,
        "model_id": config.model.model_id,
        "processed_dir": str(config.data.processed_dir),
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": config.model.max_length,
        "load_in_4bit": config.model.load_in_4bit,
        "compute_dtype": str(compute_dtype),
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "packing": config.training.packing,
        "assistant_only_loss": config.training.assistant_only_loss,
    }
    save_json(config.run_dir / "hf_train_setup.json", metadata)
    trainer.model.print_trainable_parameters()

    checkpoint = resume_from_checkpoint
    if checkpoint is None:
        checkpoint = get_last_checkpoint(str(config.checkpoints_dir))

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(config.adapter_dir))
    tokenizer.save_pretrained(config.adapter_dir)
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    save_json(config.run_dir / "hf_train_metrics.json", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QLoRA SFT with TRL.")
    parser.add_argument("--config", required=True, help="Path to HF TOML config.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path. By default, resume from the latest checkpoint if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hf_config(args.config)
    train(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
