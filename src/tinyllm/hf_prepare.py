from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from datasets import DatasetDict, load_dataset

from tinyllm.chat import build_chat_transcript, normalize_chat_role
from tinyllm.hf_config import HFExperimentConfig, load_hf_config
from tinyllm.utils import ensure_dir, save_json, set_seed


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


def _select_limit(dataset, limit: int):
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.select(range(limit))


def _create_base_splits(raw: DatasetDict, config: HFExperimentConfig) -> DatasetDict:
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
    test_fraction = config.data.test_ratio / holdout_ratio
    validation_test = holdout.train_test_split(
        test_size=test_fraction,
        seed=config.run.seed,
    )
    return DatasetDict(
        train=base_split["train"],
        validation=validation_test["train"],
        test=validation_test["test"],
    )


def _trim_messages(messages: list[dict], max_messages: int) -> list[dict]:
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages

    first_message = messages[0]
    if first_message["role"] == "system" and max_messages > 1:
        return [first_message, *messages[-(max_messages - 1) :]]
    return messages[-max_messages:]


def _format_example(example: dict, config: HFExperimentConfig) -> dict:
    conversation = example.get("conversation") or []
    messages: list[dict] = []
    for item in conversation:
        role = normalize_chat_role(item.get("role", ""))
        if role is None:
            continue
        content = _clean_chat_content(item.get("content", ""))
        if config.data.max_chars_per_message > 0:
            content = content[: config.data.max_chars_per_message].strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    messages = _trim_messages(messages, config.data.max_messages_per_example)
    text = build_chat_transcript(messages)
    return {"messages": messages, "text": text}


def _filter_example(item: dict, config: HFExperimentConfig) -> bool:
    messages = item.get("messages", [])
    if len(messages) < config.data.min_messages:
        return False
    if len(item.get("text", "")) < config.data.min_text_chars:
        return False
    if config.data.max_total_chars > 0 and len(item["text"]) > config.data.max_total_chars:
        return False
    return any(message["role"] == "assistant" for message in messages)


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


def summarize_dataset(dataset: DatasetDict) -> dict:
    summary = {}
    for split_name, split in dataset.items():
        lengths = [len(text) for text in split["text"]]
        message_counts = [len(messages) for messages in split["messages"]]
        summary[split_name] = {
            "num_examples": len(split),
            "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max_chars": max(lengths) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "avg_messages": round(sum(message_counts) / len(message_counts), 2) if message_counts else 0.0,
            "max_messages": max(message_counts) if message_counts else 0,
        }
    return summary


def build_processed_dataset(config: HFExperimentConfig) -> tuple[DatasetDict, dict]:
    set_seed(config.run.seed)
    raw = load_dataset(config.data.dataset_id)
    dataset = _create_base_splits(raw, config)
    formatted = dataset.map(
        lambda item: _format_example(item, config),
        remove_columns=dataset["train"].column_names,
        desc="Formatting chat examples",
    )
    filtered = formatted.filter(
        lambda item: _filter_example(item, config),
        desc="Filtering chat examples",
    )

    dedup_stats = {"train": 0, "validation": 0, "test": 0}
    if config.data.deduplicate:
        for split_name in filtered:
            filtered[split_name], dedup_stats[split_name] = _deduplicate_split(filtered[split_name])

    filtered["train"] = _select_limit(filtered["train"], config.data.max_train_samples)
    filtered["validation"] = _select_limit(filtered["validation"], config.data.max_validation_samples)
    filtered["test"] = _select_limit(filtered["test"], config.data.max_test_samples)
    return filtered, dedup_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare conversational dataset for HF SFT.")
    parser.add_argument("--config", required=True, help="Path to HF TOML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hf_config(args.config)
    processed_dir = Path(config.data.processed_dir)
    ensure_dir(processed_dir.parent)
    ensure_dir(config.run_dir)

    dataset, dedup_stats = build_processed_dataset(config)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    dataset.save_to_disk(str(processed_dir))

    metadata = {
        "dataset_id": config.data.dataset_id,
        "model_id": config.model.model_id,
        "processed_dir": str(processed_dir),
        "summary": summarize_dataset(dataset),
        "deduplicate": config.data.deduplicate,
        "duplicates_removed": dedup_stats,
        "max_messages_per_example": config.data.max_messages_per_example,
        "max_chars_per_message": config.data.max_chars_per_message,
        "max_total_chars": config.data.max_total_chars,
    }
    save_json(processed_dir / "metadata.json", metadata)
    save_json(config.run_dir / "dataset_summary.json", metadata)
    print(f"Saved processed dataset to {processed_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
