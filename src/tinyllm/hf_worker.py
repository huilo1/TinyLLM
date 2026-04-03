from __future__ import annotations

import argparse
import json
import traceback

from tinyllm.hf_generate import load_generator_bundle, render_messages, render_raw_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent stdio worker for HF adapter inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_generator_bundle(config_path=args.config, adapter_path=args.adapter)
    print(json.dumps({"status": "ready"}, ensure_ascii=False), flush=True)

    for raw_line in iter(input, ""):
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            kind = request.get("kind", "messages")
            max_new_tokens = int(request.get("max_new_tokens", 256))
            temperature = float(request.get("temperature", 0.7))
            top_p = float(request.get("top_p", 0.9))
            top_k = int(request.get("top_k", 0))

            if kind == "messages":
                result = render_messages(
                    bundle=bundle,
                    messages=request["messages"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            elif kind == "raw_prompt":
                result = render_raw_prompt(
                    bundle=bundle,
                    raw_prompt=str(request["raw_prompt"]),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                raise ValueError(f"Unsupported request kind: {kind}")

            print(
                json.dumps(
                    {
                        "ok": True,
                        "prompt_text": result.prompt_text,
                        "completion": result.completion,
                        "full_text": result.full_text,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except EOFError:
            break
        except Exception as exc:  # pragma: no cover - defensive path for remote worker
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
