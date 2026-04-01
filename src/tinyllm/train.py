from __future__ import annotations

import argparse
import contextlib
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tinyllm.config import ExperimentConfig, load_config
from tinyllm.data import create_block_dataset, load_tokenizer
from tinyllm.model import TinyGPT, causal_lm_loss, model_size_millions
from tinyllm.utils import count_parameters, ensure_dir, save_json, select_device, set_seed


def create_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias"):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
    )


def create_autocast(device: str):
    if device != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def create_grad_scaler(device: str) -> torch.amp.GradScaler | None:
    if device != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return None
    return torch.amp.GradScaler("cuda")


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = causal_lm_loss(logits, y)
        total_loss += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, perplexity


def save_checkpoint(
    checkpoint_path: str,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config_name": config.run.name,
        },
        checkpoint_path,
    )


def train(config: ExperimentConfig) -> None:
    set_seed(config.run.seed)
    device = select_device()
    ensure_dir(config.checkpoints_dir)
    ensure_dir(config.cache_dir)

    tokenizer = load_tokenizer(config.tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    model = TinyGPT(config.model, vocab_size=vocab_size).to(device)
    optimizer = create_optimizer(model, config)

    train_dataset = create_block_dataset(config, "train")
    val_dataset = create_block_dataset(config, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    metadata = {
        "device": device,
        "train_batches": len(train_loader),
        "validation_batches": len(val_loader),
        "vocab_size": vocab_size,
        "parameter_count": count_parameters(model),
        "estimated_model_size_millions": model_size_millions(config.model, vocab_size),
    }
    save_json(config.run_dir / "train_setup.json", metadata)
    print(metadata)

    history = []
    best_val_loss = float("inf")
    grad_scaler = create_grad_scaler(device)

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.num_epochs}")
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with create_autocast(device):
                logits = model(x)
                loss = causal_lm_loss(logits, y)
            if grad_scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss, val_ppl = evaluate(model, val_loader, device=device)
        epoch_seconds = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "validation_perplexity": round(val_ppl, 4) if math.isfinite(val_ppl) else "inf",
            "epoch_seconds": round(epoch_seconds, 2),
        }
        history.append(epoch_record)
        save_json(config.run_dir / "history.json", {"epochs": history})

        last_checkpoint = config.checkpoints_dir / "last.pt"
        save_checkpoint(str(last_checkpoint), model, optimizer, config, epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = config.checkpoints_dir / "best.pt"
            save_checkpoint(str(best_checkpoint), model, optimizer, config, epoch, best_val_loss)

        print(epoch_record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny decoder-only LM from scratch.")
    parser.add_argument(
        "--config",
        default="configs/tiny_rfn.toml",
        help="Path to TOML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
