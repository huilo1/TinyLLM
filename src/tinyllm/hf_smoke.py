from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.hf_generate import load_generator_bundle, render_completion


def _load_prompts(path: str | Path, limit: int) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append(item)
            if len(prompts) >= limit:
                break
    return prompts


def run_smoke(
    config_path: str,
    adapter_path: str,
    prompts_file: str,
    output_path: str | None,
    limit: int,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, object]]:
    bundle = load_generator_bundle(config_path=config_path, adapter_path=adapter_path)
    prompts = _load_prompts(prompts_file, limit=limit)

    results: list[dict[str, object]] = []
    for item in prompts:
        prompt = str(item["prompt"]).strip()
        completion = render_completion(
            bundle=bundle,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": prompt,
                "watch": item.get("watch"),
                "response": completion,
            }
        )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick prompt smoke-test against a QLoRA adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompts-file", default="data/evals/base-vs-lora-prompts.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--system-prompt", default="Ты полезный русскоязычный ассистент. Отвечай ясно и по делу.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_smoke(
        config_path=args.config,
        adapter_path=args.adapter,
        prompts_file=args.prompts_file,
        output_path=args.output,
        limit=args.limit,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(json.dumps({"samples": len(results), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
