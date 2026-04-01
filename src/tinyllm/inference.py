from __future__ import annotations

from pathlib import Path

import torch

from tinyllm.config import ExperimentConfig, load_config
from tinyllm.data import load_tokenizer
from tinyllm.model import TinyGPT
from tinyllm.utils import select_device


def build_news_prompt(
    source: str,
    date: str,
    title: str,
    body_prefix: str = "",
) -> str:
    source = source.strip() or "finam"
    date = date.strip() or "2024-01-15"
    title = title.strip()
    body_prefix = body_prefix.strip()

    parts = [
        f"Источник: {source}",
        f"Дата: {date}",
    ]
    if title:
        parts.append(f"Заголовок: {title}")

    text_line = "Текст:"
    if body_prefix:
        text_line = f"{text_line} {body_prefix}"
    parts.append(text_line)
    return "\n".join(parts)


class TinyLLMGenerator:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.device = device or select_device()
        self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        self.model = TinyGPT(
            self.config.model,
            vocab_size=self.tokenizer.get_vocab_size(),
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.config.generation.max_new_tokens
        temperature = temperature or self.config.generation.temperature
        top_k = top_k if top_k is not None else self.config.generation.top_k

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        input_ids = torch.tensor(
            [[self.bos_id, *prompt_ids]],
            dtype=torch.long,
            device=self.device,
        )

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.eos_id,
        )
        generated_ids = output_ids[0].tolist()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def complete(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> tuple[str, str]:
        full_text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        if full_text.startswith(prompt):
            completion = full_text[len(prompt) :].lstrip()
        else:
            completion = full_text
        return completion, full_text


def load_generator(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | None = None,
) -> TinyLLMGenerator:
    return TinyLLMGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
