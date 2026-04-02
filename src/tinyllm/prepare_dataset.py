from __future__ import annotations

import argparse
from pathlib import Path
import re

from datasets import DatasetDict, load_dataset

from tinyllm.chat import build_chat_transcript, normalize_chat_role
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


def _clean_chat_content(value: str) -> str:
    value = str(value or "")
    value = value.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _dedup_key(text: str) -> str:
    key = text.lower()
    key = re.sub(r"\W+", " ", key)
    return key.strip()


def _format_news_example(example: dict, config: ExperimentConfig) -> dict:
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


def _format_chat_example(example: dict, config: ExperimentConfig) -> dict:
    conversation = example.get("conversation") or []
    messages = []
    for item in conversation:
        role = normalize_chat_role(item.get("role", ""))
        if role is None:
            continue
        content = _clean_chat_content(item.get("content", ""))
        if not content:
            continue
        messages.append({"role": role, "content": content})

    return {
        "text": build_chat_transcript(messages, separator=config.data.text_separator),
        "messages": messages,
    }


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


def _create_base_splits(raw: DatasetDict, config: ExperimentConfig) -> DatasetDict:
    train_split_name = "train" if "train" in raw else next(iter(raw.keys()))

    if "test" in raw:
        split = raw[train_split_name].train_test_split(
            test_size=config.data.validation_ratio,
            seed=config.run.seed,
        )
        return DatasetDict(
            train=split["train"],
            validation=split["test"],
            test=raw["test"],
        )

    holdout_ratio = config.data.validation_ratio + config.data.test_ratio
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("validation_ratio + test_ratio must be between 0 and 1 when dataset has no test split.")

    base_split = raw[train_split_name].train_test_split(
        test_size=holdout_ratio,
        seed=config.run.seed,
    )
    holdout = base_split["test"]

    if config.data.test_ratio <= 0:
        validation = holdout
        test = holdout.select([])
    elif config.data.validation_ratio <= 0:
        validation = holdout.select([])
        test = holdout
    else:
        test_fraction = config.data.test_ratio / holdout_ratio
        validation_test = holdout.train_test_split(
            test_size=test_fraction,
            seed=config.run.seed,
        )
        validation = validation_test["train"]
        test = validation_test["test"]

    return DatasetDict(
        train=base_split["train"],
        validation=validation,
        test=test,
    )


def _filter_chat_example(item: dict, config: ExperimentConfig) -> bool:
    messages = item.get("messages", [])
    if len(messages) < config.data.min_messages:
        return False
    if len(item.get("text", "")) < config.data.min_text_chars:
        return False
    return any(message["role"] == "assistant" for message in messages)


def build_processed_dataset(config: ExperimentConfig) -> tuple[DatasetDict, dict]:
    set_seed(config.run.seed)
    raw = load_dataset(config.data.dataset_id)
    dataset = _create_base_splits(raw, config)
    formatter = _format_chat_example if config.is_chat_model else _format_news_example

    formatted = dataset.map(
        lambda item: formatter(item, config),
        remove_columns=dataset["train"].column_names,
        desc="Formatting texts",
    )
    if config.is_chat_model:
        filtered = formatted.filter(
            lambda item: _filter_chat_example(item, config),
            desc="Filtering chat conversations",
        )
    else:
        filtered = formatted.filter(
            lambda item: len(item["text"]) >= config.data.min_text_chars,
            desc="Filtering short texts",
        )

    dedup_stats = {"train": 0, "validation": 0, "test": 0}
    if config.data.deduplicate:
        for split_name in filtered:
            filtered[split_name], dedup_stats[split_name] = _deduplicate_split(filtered[split_name])

    filtered["train"] = _select_limit(filtered["train"], config.data.max_train_samples)
    filtered["validation"] = _select_limit(filtered["validation"], config.data.max_validation_samples)
    filtered["test"] = _select_limit(filtered["test"], config.data.max_test_samples)
    return filtered, dedup_stats


def summarize_dataset(dataset: DatasetDict) -> dict:
    summary = {}
    for split_name, split in dataset.items():
        lengths = [len(text) for text in split["text"]]
        split_summary = {
            "num_examples": len(split),
            "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max_chars": max(lengths) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
        }
        if "messages" in split.column_names:
            message_counts = [len(messages) for messages in split["messages"]]
            split_summary["avg_messages"] = round(sum(message_counts) / len(message_counts), 2) if message_counts else 0.0
            split_summary["max_messages"] = max(message_counts) if message_counts else 0
        summary[split_name] = split_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training dataset.")
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
        "format": config.data.format,
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
