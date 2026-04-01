from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

from tinyllm.config import load_config
from tinyllm.utils import ensure_dir, save_json

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer(config_path: str | Path) -> Path:
    config = load_config(config_path)
    dataset = load_from_disk(str(config.data.processed_dir))
    tokenizer_dir = ensure_dir(config.tokenizer_dir)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    save_json(
        tokenizer_dir / "tokenizer_meta.json",
        {
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": SPECIAL_TOKENS,
            "train_examples": len(dataset["train"]),
        },
    )
    return tokenizer_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tokenizer from scratch.")
    parser.add_argument(
        "--config",
        default="configs/tiny_rfn.toml",
        help="Path to TOML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = train_tokenizer(args.config)
    print(f"Saved tokenizer to {tokenizer_path}")


if __name__ == "__main__":
    main()

