from __future__ import annotations

import argparse

from tinyllm.inference import load_generator


def generate_text(config_path: str, checkpoint_path: str, prompt: str) -> str:
    generator = load_generator(config_path, checkpoint_path)
    return generator.generate(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--config", default="configs/tiny_rfn.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = generate_text(args.config, args.checkpoint, args.prompt)
    print(text)


if __name__ == "__main__":
    main()
