from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class HFRunConfig:
    name: str
    seed: int
    artifacts_dir: Path


@dataclass(slots=True)
class HFDataConfig:
    dataset_id: str
    processed_dir: Path
    validation_ratio: float
    test_ratio: float
    min_text_chars: int
    min_messages: int
    max_train_samples: int
    max_validation_samples: int
    max_test_samples: int
    deduplicate: bool
    max_messages_per_example: int
    max_chars_per_message: int
    max_total_chars: int


@dataclass(slots=True)
class HFModelConfig:
    model_id: str
    max_length: int
    load_in_4bit: bool
    use_bf16: bool
    attn_implementation: str


@dataclass(slots=True)
class HFLoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: str | list[str]


@dataclass(slots=True)
class HFTrainingConfig:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    eval_strategy: str
    eval_steps: int
    save_total_limit: int
    lr_scheduler_type: str
    gradient_checkpointing: bool
    assistant_only_loss: bool
    packing: bool
    max_steps: int


@dataclass(slots=True)
class HFExperimentConfig:
    run: HFRunConfig
    data: HFDataConfig
    model: HFModelConfig
    lora: HFLoraConfig
    training: HFTrainingConfig

    @property
    def run_dir(self) -> Path:
        return self.run.artifacts_dir / self.run.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"


def _load_target_modules(value: str | list[str]) -> str | list[str]:
    if isinstance(value, str):
        return value
    return [str(item) for item in value]


def load_hf_config(config_path: str | Path) -> HFExperimentConfig:
    path = Path(config_path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = HFRunConfig(
        name=raw["run"]["name"],
        seed=int(raw["run"]["seed"]),
        artifacts_dir=Path(raw["run"]["artifacts_dir"]),
    )
    data_raw = raw["data"]
    data = HFDataConfig(
        dataset_id=data_raw["dataset_id"],
        processed_dir=Path(data_raw["processed_dir"]),
        validation_ratio=float(data_raw["validation_ratio"]),
        test_ratio=float(data_raw.get("test_ratio", data_raw["validation_ratio"])),
        min_text_chars=int(data_raw.get("min_text_chars", 0)),
        min_messages=int(data_raw.get("min_messages", 2)),
        max_train_samples=int(data_raw.get("max_train_samples", 0)),
        max_validation_samples=int(data_raw.get("max_validation_samples", 0)),
        max_test_samples=int(data_raw.get("max_test_samples", 0)),
        deduplicate=bool(data_raw.get("deduplicate", True)),
        max_messages_per_example=int(data_raw.get("max_messages_per_example", 12)),
        max_chars_per_message=int(data_raw.get("max_chars_per_message", 4000)),
        max_total_chars=int(data_raw.get("max_total_chars", 12000)),
    )
    model_raw = raw["model"]
    model = HFModelConfig(
        model_id=model_raw["model_id"],
        max_length=int(model_raw.get("max_length", 1024)),
        load_in_4bit=bool(model_raw.get("load_in_4bit", True)),
        use_bf16=bool(model_raw.get("use_bf16", True)),
        attn_implementation=str(model_raw.get("attn_implementation", "sdpa")),
    )
    lora_raw = raw["lora"]
    lora = HFLoraConfig(
        r=int(lora_raw.get("r", 16)),
        alpha=int(lora_raw.get("alpha", 32)),
        dropout=float(lora_raw.get("dropout", 0.05)),
        target_modules=_load_target_modules(lora_raw.get("target_modules", "all-linear")),
    )
    train_raw = raw["training"]
    training = HFTrainingConfig(
        per_device_train_batch_size=int(train_raw.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_raw.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_raw.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_raw.get("learning_rate", 2e-4)),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        num_train_epochs=float(train_raw.get("num_train_epochs", 2.0)),
        warmup_ratio=float(train_raw.get("warmup_ratio", 0.03)),
        logging_steps=int(train_raw.get("logging_steps", 10)),
        save_steps=int(train_raw.get("save_steps", 200)),
        eval_strategy=str(train_raw.get("eval_strategy", "steps")),
        eval_steps=int(train_raw.get("eval_steps", 200)),
        save_total_limit=int(train_raw.get("save_total_limit", 2)),
        lr_scheduler_type=str(train_raw.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(train_raw.get("gradient_checkpointing", True)),
        assistant_only_loss=bool(train_raw.get("assistant_only_loss", False)),
        packing=bool(train_raw.get("packing", False)),
        max_steps=int(train_raw.get("max_steps", -1)),
    )
    return HFExperimentConfig(
        run=run,
        data=data,
        model=model,
        lora=lora,
        training=training,
    )
