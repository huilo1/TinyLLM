from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class RunConfig:
    name: str
    seed: int
    artifacts_dir: Path


@dataclass(slots=True)
class DataConfig:
    dataset_id: str
    processed_dir: Path
    validation_ratio: float
    min_body_chars: int
    max_train_samples: int
    max_validation_samples: int
    max_test_samples: int
    text_separator: str
    deduplicate: bool


@dataclass(slots=True)
class TokenizerConfig:
    vocab_size: int
    min_frequency: int


@dataclass(slots=True)
class ModelConfig:
    sequence_length: int
    n_layers: int
    n_heads: int
    d_model: int
    mlp_ratio: int
    dropout: float


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    grad_clip: float
    num_workers: int
    warmup_ratio: float
    min_lr_ratio: float


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int
    sample_prompts: list[str]


@dataclass(slots=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig

    @property
    def run_dir(self) -> Path:
        return self.run.artifacts_dir / self.run.name

    @property
    def tokenizer_dir(self) -> Path:
        return self.run_dir / "tokenizer"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / "cache"


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = RunConfig(
        name=raw["run"]["name"],
        seed=int(raw["run"]["seed"]),
        artifacts_dir=Path(raw["run"]["artifacts_dir"]),
    )
    data = DataConfig(
        dataset_id=raw["data"]["dataset_id"],
        processed_dir=Path(raw["data"]["processed_dir"]),
        validation_ratio=float(raw["data"]["validation_ratio"]),
        min_body_chars=int(raw["data"]["min_body_chars"]),
        max_train_samples=int(raw["data"]["max_train_samples"]),
        max_validation_samples=int(raw["data"]["max_validation_samples"]),
        max_test_samples=int(raw["data"]["max_test_samples"]),
        text_separator=raw["data"]["text_separator"],
        deduplicate=bool(raw["data"].get("deduplicate", False)),
    )
    tokenizer = TokenizerConfig(
        vocab_size=int(raw["tokenizer"]["vocab_size"]),
        min_frequency=int(raw["tokenizer"]["min_frequency"]),
    )
    model = ModelConfig(
        sequence_length=int(raw["model"]["sequence_length"]),
        n_layers=int(raw["model"]["n_layers"]),
        n_heads=int(raw["model"]["n_heads"]),
        d_model=int(raw["model"]["d_model"]),
        mlp_ratio=int(raw["model"]["mlp_ratio"]),
        dropout=float(raw["model"]["dropout"]),
    )
    training = TrainingConfig(
        batch_size=int(raw["training"]["batch_size"]),
        eval_batch_size=int(raw["training"]["eval_batch_size"]),
        learning_rate=float(raw["training"]["learning_rate"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        num_epochs=int(raw["training"]["num_epochs"]),
        grad_clip=float(raw["training"]["grad_clip"]),
        num_workers=int(raw["training"]["num_workers"]),
        warmup_ratio=float(raw["training"].get("warmup_ratio", 0.0)),
        min_lr_ratio=float(raw["training"].get("min_lr_ratio", 1.0)),
    )
    generation = GenerationConfig(
        max_new_tokens=int(raw["generation"]["max_new_tokens"]),
        temperature=float(raw["generation"]["temperature"]),
        top_k=int(raw["generation"]["top_k"]),
        sample_prompts=list(raw["generation"].get("sample_prompts", [])),
    )
    return ExperimentConfig(
        run=run,
        data=data,
        tokenizer=tokenizer,
        model=model,
        training=training,
        generation=generation,
    )
