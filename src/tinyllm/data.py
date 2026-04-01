from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tinyllm.config import ExperimentConfig
from tinyllm.utils import ensure_dir


class TokenBlockDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, sequence_length: int):
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        usable_tokens = self.tokens.numel() - 1
        return max(usable_tokens // self.sequence_length, 0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        chunk = self.tokens[start : start + self.sequence_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run tinyllm-train-tokenizer first."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def _encode_texts(tokenizer: Tokenizer, texts: list[str]) -> torch.Tensor:
    token_ids: list[int] = []
    batch_size = 512
    for offset in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[offset : offset + batch_size]
        encoded = tokenizer.encode_batch(batch)
        for item in encoded:
            token_ids.extend(item.ids)
    return torch.tensor(token_ids, dtype=torch.long)


def _cache_path(config: ExperimentConfig, split: str) -> Path:
    return config.cache_dir / f"{split}_tokens.pt"


def load_or_create_tokens(config: ExperimentConfig, split: str) -> torch.Tensor:
    cache_path = _cache_path(config, split)
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    dataset = load_from_disk(str(config.data.processed_dir))
    tokenizer = load_tokenizer(config.tokenizer_dir)
    tokens = _encode_texts(tokenizer, dataset[split]["text"])
    ensure_dir(cache_path.parent)
    torch.save(tokens, cache_path)
    return tokens


def create_block_dataset(config: ExperimentConfig, split: str) -> TokenBlockDataset:
    tokens = load_or_create_tokens(config, split)
    return TokenBlockDataset(tokens=tokens, sequence_length=config.model.sequence_length)

