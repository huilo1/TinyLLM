from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tinyllm.chat import render_chat_message
from tinyllm.config import ExperimentConfig
from tinyllm.utils import ensure_dir


class TokenBlockDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, sequence_length: int, loss_mask: torch.Tensor | None = None):
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.loss_mask = loss_mask

    def __len__(self) -> int:
        usable_tokens = self.tokens.numel() - 1
        return max(usable_tokens // self.sequence_length, 0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        start = index * self.sequence_length
        chunk = self.tokens[start : start + self.sequence_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        if self.loss_mask is None:
            return x, y

        mask_chunk = self.loss_mask[start : start + self.sequence_length + 1]
        return x, y, mask_chunk[1:]


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


def _encode_chat_messages(
    tokenizer: Tokenizer,
    messages_batch: list[list[dict]],
    separator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    separator_ids = tokenizer.encode(separator, add_special_tokens=False).ids if separator else []

    token_ids: list[int] = []
    loss_mask: list[bool] = []
    for messages in tqdm(messages_batch, desc="Encoding chat conversations"):
        if not messages:
            continue

        token_ids.append(bos_id)
        loss_mask.append(False)

        for index, message in enumerate(messages):
            rendered = render_chat_message(message["role"], message["content"])
            encoded = tokenizer.encode(rendered, add_special_tokens=False).ids
            is_assistant = message["role"] == "assistant"
            token_ids.extend(encoded)
            loss_mask.extend([is_assistant] * len(encoded))

            if index < len(messages) - 1 and separator_ids:
                token_ids.extend(separator_ids)
                loss_mask.extend([is_assistant] * len(separator_ids))

        token_ids.append(eos_id)
        loss_mask.append(messages[-1]["role"] == "assistant")

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(loss_mask, dtype=torch.bool)


def _cache_path(config: ExperimentConfig, split: str) -> Path:
    return config.cache_dir / f"{split}_tokens.pt"


def _cache_mask_path(config: ExperimentConfig, split: str) -> Path:
    return config.cache_dir / f"{split}_loss_mask.pt"


def load_or_create_training_tensors(
    config: ExperimentConfig,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cache_path = _cache_path(config, split)
    mask_path = _cache_mask_path(config, split)
    if cache_path.exists():
        if not config.is_chat_model:
            return torch.load(cache_path, weights_only=False), None
        if mask_path.exists():
            tokens = torch.load(cache_path, weights_only=False)
            loss_mask = torch.load(mask_path, weights_only=False)
            return tokens, loss_mask
        cache_path.unlink()

    dataset = load_from_disk(str(config.data.processed_dir))
    tokenizer = load_tokenizer(config.tokenizer_dir)
    loss_mask = None
    if config.is_chat_model:
        tokens, loss_mask = _encode_chat_messages(
            tokenizer=tokenizer,
            messages_batch=dataset[split]["messages"],
            separator=config.data.text_separator,
        )
    else:
        tokens = _encode_texts(tokenizer, dataset[split]["text"])

    ensure_dir(cache_path.parent)
    torch.save(tokens, cache_path)
    if loss_mask is not None:
        torch.save(loss_mask, mask_path)
    return tokens, loss_mask


def load_or_create_tokens(config: ExperimentConfig, split: str) -> torch.Tensor:
    tokens, _ = load_or_create_training_tensors(config, split)
    return tokens


def create_block_dataset(config: ExperimentConfig, split: str) -> TokenBlockDataset:
    tokens, loss_mask = load_or_create_training_tensors(config, split)
    return TokenBlockDataset(
        tokens=tokens,
        sequence_length=config.model.sequence_length,
        loss_mask=loss_mask,
    )
