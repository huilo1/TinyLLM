from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from tinyllm.config import load_config
from tinyllm.data import create_block_dataset, load_tokenizer
from tinyllm.model import TinyGPT, causal_lm_loss
from tinyllm.utils import save_json, select_device


@torch.no_grad()
def run_evaluation(config_path: str, checkpoint_path: str, split: str) -> dict:
    config = load_config(config_path)
    device = select_device()
    tokenizer = load_tokenizer(config.tokenizer_dir)
    model = TinyGPT(config.model, vocab_size=tokenizer.get_vocab_size()).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = create_block_dataset(config, split)
    loader = DataLoader(
        dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        loss_mask = batch[2].to(device) if len(batch) > 2 else None
        logits = model(x)
        loss = causal_lm_loss(logits, y, loss_mask=loss_mask)
        total_loss += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    result = {
        "split": split,
        "loss": round(mean_loss, 4),
        "perplexity": round(perplexity, 4) if math.isfinite(perplexity) else "inf",
    }
    save_json(config.run_dir / f"eval_{split}.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/tiny_rfn.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(args.config, args.checkpoint, args.split)
    print(result)


if __name__ == "__main__":
    main()
