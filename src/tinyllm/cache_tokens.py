from __future__ import annotations

import argparse

from tinyllm.config import load_config
from tinyllm.data import load_or_create_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize token caches before training.")
    parser.add_argument("--config", default="configs/tiny_rfn.toml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Dataset splits to tokenize and cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for split in args.splits:
        tokens = load_or_create_tokens(config, split)
        print(f"{split}: cached {tokens.numel()} tokens")


if __name__ == "__main__":
    main()
