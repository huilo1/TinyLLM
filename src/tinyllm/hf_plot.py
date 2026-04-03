from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tinyllm.hf_config import load_hf_config
from tinyllm.utils import ensure_dir, save_json

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def _parse_step_from_checkpoint(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.as_posix())
    if match is None:
        return -1
    return int(match.group(1))


def _coerce_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def _resolve_run_dir(config_path: str | None, run_dir: str | None) -> Path:
    if config_path:
        return load_hf_config(config_path).run_dir
    if run_dir:
        return Path(run_dir)
    raise ValueError("Either --config or --run-dir must be provided.")


def _find_latest_trainer_state(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("checkpoints/checkpoint-*/trainer_state.json"))
    if not candidates:
        direct_state = run_dir / "checkpoints" / "trainer_state.json"
        if direct_state.exists():
            return direct_state
        raise FileNotFoundError(
            f"No trainer_state.json found under {run_dir / 'checkpoints'}. "
            "Run must have at least one saved checkpoint."
        )
    return max(candidates, key=_parse_step_from_checkpoint)


def _load_run_metadata(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "hf_train_setup.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_history(trainer_state_path: Path) -> tuple[list[dict], dict[str, object]]:
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"Unexpected log_history format in {trainer_state_path}.")
    return history, payload


def _extract_train_rows(history: list[dict]) -> list[dict[str, float | int | None]]:
    rows = []
    for item in history:
        if "loss" not in item or "step" not in item:
            continue
        step = int(item["step"])
        rows.append(
            {
                "step": step,
                "epoch": _coerce_float(item.get("epoch")),
                "loss": _coerce_float(item.get("loss")),
                "entropy": _coerce_float(item.get("entropy")),
                "mean_token_accuracy": _coerce_float(item.get("mean_token_accuracy")),
                "learning_rate": _coerce_float(item.get("learning_rate")),
                "grad_norm": _coerce_float(item.get("grad_norm")),
                "num_tokens": _coerce_float(item.get("num_tokens")),
            }
        )
    return rows


def _extract_eval_rows(history: list[dict]) -> list[dict[str, float | int | None]]:
    rows = []
    for item in history:
        if "eval_loss" not in item or "step" not in item:
            continue
        step = int(item["step"])
        rows.append(
            {
                "step": step,
                "epoch": _coerce_float(item.get("epoch")),
                "eval_loss": _coerce_float(item.get("eval_loss")),
                "eval_entropy": _coerce_float(item.get("eval_entropy")),
                "eval_mean_token_accuracy": _coerce_float(item.get("eval_mean_token_accuracy")),
                "eval_runtime": _coerce_float(item.get("eval_runtime")),
                "eval_samples_per_second": _coerce_float(item.get("eval_samples_per_second")),
                "eval_steps_per_second": _coerce_float(item.get("eval_steps_per_second")),
                "eval_num_tokens": _coerce_float(item.get("eval_num_tokens")),
            }
        )
    return rows


def _moving_average(values: list[float | None], window: int) -> list[float | None]:
    window = max(window, 1)
    smoothed: list[float | None] = []
    active: list[float] = []
    for value in values:
        if value is None:
            smoothed.append(None)
            continue
        active.append(value)
        if len(active) > window:
            active.pop(0)
        smoothed.append(sum(active) / len(active))
    return smoothed


def _write_csv(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_title(run_dir: Path, metadata: dict[str, object], override: str | None) -> str:
    if override:
        return override
    run_name = str(metadata.get("run_name", run_dir.name))
    model_id = metadata.get("model_id")
    if model_id:
        return f"{run_name} • {model_id}"
    return run_name


def _plot_history(
    train_rows: list[dict[str, float | int | None]],
    eval_rows: list[dict[str, float | int | None]],
    output_prefix: Path,
    title: str,
    smoothing_window: int,
) -> None:
    if not train_rows:
        raise ValueError("No train loss points found in trainer_state.json.")

    train_steps = [int(row["step"]) for row in train_rows]
    train_loss = [row["loss"] for row in train_rows]
    train_accuracy = [row["mean_token_accuracy"] for row in train_rows]
    learning_rates = [row["learning_rate"] for row in train_rows]

    smoothed_loss = _moving_average(train_loss, smoothing_window)
    smoothed_accuracy = _moving_average(train_accuracy, smoothing_window)

    eval_steps = [int(row["step"]) for row in eval_rows]
    eval_loss = [row["eval_loss"] for row in eval_rows]
    eval_accuracy = [row["eval_mean_token_accuracy"] for row in eval_rows]

    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "axes.titlecolor": "#0f172a",
            "font.size": 11,
        }
    )

    fig, (ax_loss, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.2]},
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax_loss.plot(train_steps, train_loss, color="#94a3b8", linewidth=1.0, alpha=0.35, label="Train loss (raw)")
    ax_loss.plot(
        train_steps,
        smoothed_loss,
        color="#2563eb",
        linewidth=2.5,
        label=f"Train loss (moving avg {smoothing_window})",
    )
    if eval_rows:
        ax_loss.plot(
            eval_steps,
            eval_loss,
            color="#ea580c",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Eval loss",
        )
        latest_eval_step = eval_steps[-1]
        latest_eval_loss = eval_loss[-1]
        if latest_eval_loss is not None:
            ax_loss.annotate(
                f"latest eval\nstep {latest_eval_step}\nloss {latest_eval_loss:.4f}",
                xy=(latest_eval_step, latest_eval_loss),
                xytext=(12, 10),
                textcoords="offset points",
                fontsize=10,
                color="#9a3412",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ed", "edgecolor": "#fdba74"},
            )

    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    ax_loss.legend(loc="upper right", frameon=False)

    ax_acc.plot(
        train_steps,
        smoothed_accuracy,
        color="#0891b2",
        linewidth=2.0,
        label=f"Train token accuracy (moving avg {smoothing_window})",
    )
    if eval_rows:
        ax_acc.plot(
            eval_steps,
            eval_accuracy,
            color="#16a34a",
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            label="Eval token accuracy",
        )
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Step")
    ax_acc.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    ax_acc.legend(loc="lower right", frameon=False)

    lr_min = min(lr for lr in learning_rates if lr is not None)
    lr_max = max(lr for lr in learning_rates if lr is not None)
    ax_acc.text(
        0.01,
        0.04,
        f"learning rate range: {lr_min:.2e} .. {lr_max:.2e}",
        transform=ax_acc.transAxes,
        fontsize=10,
        color="#475569",
    )

    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    ensure_dir(png_path.parent)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HF QLoRA training curves from trainer_state.json.")
    parser.add_argument("--config", default=None, help="Optional HF TOML config to resolve the run directory.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing checkpoints/ and hf_train_setup.json.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots and CSVs. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=25,
        help="Moving-average window for train curves.",
    )
    parser.add_argument("--title", default=None, help="Optional chart title override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.config, args.run_dir)
    trainer_state_path = _find_latest_trainer_state(run_dir)
    metadata = _load_run_metadata(run_dir)
    history, trainer_state = _load_history(trainer_state_path)
    train_rows = _extract_train_rows(history)
    eval_rows = _extract_eval_rows(history)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    ensure_dir(output_dir)
    output_prefix = output_dir / "training_history"

    _write_csv(output_dir / "train_metrics.csv", train_rows)
    _write_csv(output_dir / "eval_metrics.csv", eval_rows)
    _plot_history(
        train_rows=train_rows,
        eval_rows=eval_rows,
        output_prefix=output_prefix,
        title=_build_title(run_dir, metadata, args.title),
        smoothing_window=args.smoothing_window,
    )

    summary = {
        "run_dir": str(run_dir),
        "trainer_state_path": str(trainer_state_path),
        "latest_step": trainer_state.get("global_step"),
        "train_points": len(train_rows),
        "eval_points": len(eval_rows),
        "latest_eval_loss": eval_rows[-1]["eval_loss"] if eval_rows else None,
        "latest_eval_accuracy": eval_rows[-1]["eval_mean_token_accuracy"] if eval_rows else None,
        "smoothing_window": args.smoothing_window,
        "output_png": str(output_prefix.with_suffix(".png")),
        "output_svg": str(output_prefix.with_suffix(".svg")),
    }
    save_json(output_dir / "plot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
