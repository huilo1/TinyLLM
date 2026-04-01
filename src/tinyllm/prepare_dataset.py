from __future__ import annotations

import argparse
from pathlib import Path
import re

from datasets import DatasetDict, load_dataset

from tinyllm.config import ExperimentConfig, load_config
from tinyllm.utils import ensure_dir, save_json, set_seed

BAD_TITLES = {
    "no title",
    "none",
    "null",
    "nan",
    "без заголовка",
}


def _clean_text(value: str) -> str:
    value = str(value or "")
    value = value.replace("\ufeff", " ").replace("\u200b", " ").replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.:;!?])", r"\1", value)
    return value.strip()


def _clean_title(value: str) -> str:
    title = _clean_text(value)
    if title.lower() in BAD_TITLES:
        return ""
    if len(title) < 4:
        return ""
    return title


def _clean_body(value: str) -> str:
    body = _clean_text(value)
    body = re.sub(r'^[\"\'«»„“”]+', "", body)
    body = re.sub(r'[\"\'«»„“”]+$', "", body)
    return body.strip()


def _dedup_key(text: str) -> str:
    key = text.lower()
    key = re.sub(r"\W+", " ", key)
    return key.strip()


def _format_example(example: dict, config: ExperimentConfig) -> dict:
    title = _clean_title(example["title"])
    body = _clean_body(example["body"])
    source = _clean_text(example["source"])
    date = _clean_text(example["date"])

    parts = [
        f"Источник: {source}",
        f"Дата: {date}",
    ]
    if title and title.lower() != "no title":
        parts.append(f"Заголовок: {title}")
    parts.append(f"Текст: {body}")

    return {"text": config.data.text_separator.join(parts)}


def _select_limit(dataset, limit: int):
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.select(range(limit))


def _deduplicate_split(split):
    texts = split["text"]
    seen: set[str] = set()
    keep_indices: list[int] = []
    for index, text in enumerate(texts):
        key = _dedup_key(text)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    removed = len(texts) - len(keep_indices)
    if removed == 0:
        return split, removed
    return split.select(keep_indices), removed


def build_processed_dataset(config: ExperimentConfig) -> tuple[DatasetDict, dict]:
    set_seed(config.run.seed)
    raw = load_dataset(config.data.dataset_id)

    split = raw["train"].train_test_split(
        test_size=config.data.validation_ratio,
        seed=config.run.seed,
    )
    dataset = DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )

    formatted = dataset.map(
        lambda item: _format_example(item, config),
        remove_columns=dataset["train"].column_names,
        desc="Formatting texts",
    )
    filtered = formatted.filter(
        lambda item: len(item["text"]) >= config.data.min_body_chars,
        desc="Filtering short texts",
    )

    dedup_stats = {"train": 0, "validation": 0, "test": 0}
    if config.data.deduplicate:
        for split_name in filtered:
            filtered[split_name], dedup_stats[split_name] = _deduplicate_split(filtered[split_name])

    filtered["train"] = _select_limit(filtered["train"], config.data.max_train_samples)
    filtered["validation"] = _select_limit(
        filtered["validation"],
        config.data.max_validation_samples,
    )
    filtered["test"] = _select_limit(filtered["test"], config.data.max_test_samples)
    return filtered, dedup_stats


def summarize_dataset(dataset: DatasetDict) -> dict:
    summary = {}
    for split_name, split in dataset.items():
        lengths = [len(text) for text in split["text"]]
        summary[split_name] = {
            "num_examples": len(split),
            "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max_chars": max(lengths) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RussianFinancialNews dataset.")
    parser.add_argument(
        "--config",
        default="configs/tiny_rfn.toml",
        help="Path to TOML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_dir = Path(config.data.processed_dir)
    ensure_dir(processed_dir.parent)

    dataset, dedup_stats = build_processed_dataset(config)
    if processed_dir.exists():
        import shutil

        shutil.rmtree(processed_dir)
    dataset.save_to_disk(str(processed_dir))

    metadata = {
        "dataset_id": config.data.dataset_id,
        "processed_dir": str(processed_dir),
        "summary": summarize_dataset(dataset),
        "deduplicate": config.data.deduplicate,
        "duplicates_removed": dedup_stats,
    }
    save_json(processed_dir / "metadata.json", metadata)
    save_json(config.run_dir / "dataset_summary.json", metadata)
    print(f"Saved processed dataset to {processed_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
