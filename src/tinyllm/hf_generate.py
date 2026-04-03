from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys

from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from tinyllm.hf_config import load_hf_config


@dataclass(slots=True)
class HFGeneratorBundle:
    config: object
    model: object
    tokenizer: object


@dataclass(slots=True)
class HFGenerationResult:
    prompt_text: str
    completion: str
    full_text: str


def _compute_dtype(config) -> torch.dtype:
    if config.model.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_generator_bundle(config_path: str, adapter_path: str) -> HFGeneratorBundle:
    config = load_hf_config(config_path)
    dtype = _compute_dtype(config)
    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    return HFGeneratorBundle(config=config, model=model, tokenizer=tokenizer)


def _build_generation_kwargs(
    bundle: HFGeneratorBundle,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": bundle.tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            kwargs["top_k"] = top_k
    else:
        kwargs["do_sample"] = False
    return kwargs


def render_messages(
    bundle: HFGeneratorBundle,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> HFGenerationResult:
    prompt_text = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = bundle.tokenizer(prompt_text, return_tensors="pt").to(bundle.model.device)

    outputs = bundle.model.generate(
        **inputs,
        **_build_generation_kwargs(
            bundle=bundle,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )
    completion = outputs[0][inputs["input_ids"].shape[-1] :]
    completion_text = bundle.tokenizer.decode(completion, skip_special_tokens=True).strip()
    full_text = bundle.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return HFGenerationResult(
        prompt_text=prompt_text,
        completion=completion_text,
        full_text=full_text,
    )


def render_raw_prompt(
    bundle: HFGeneratorBundle,
    raw_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> HFGenerationResult:
    inputs = bundle.tokenizer(raw_prompt, return_tensors="pt").to(bundle.model.device)
    outputs = bundle.model.generate(
        **inputs,
        **_build_generation_kwargs(
            bundle=bundle,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )
    completion = outputs[0][inputs["input_ids"].shape[-1] :]
    completion_text = bundle.tokenizer.decode(completion, skip_special_tokens=True).strip()
    full_text = bundle.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return HFGenerationResult(
        prompt_text=raw_prompt,
        completion=completion_text,
        full_text=full_text,
    )


def render_completion(
    bundle: HFGeneratorBundle,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    result = render_messages(
        bundle=bundle,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return result.completion


def generate_text(
    config_path: str,
    adapter_path: str,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> str:
    bundle = load_generator_bundle(config_path=config_path, adapter_path=adapter_path)
    return render_completion(
        bundle=bundle,
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a QLoRA adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--system-prompt", default="Ты полезный русскоязычный ассистент. Отвечай ясно и по делу.")
    parser.add_argument("--raw-prompt", default=None)
    parser.add_argument("--messages-json", default=None)
    parser.add_argument("--messages-stdin", action="store_true")
    parser.add_argument("--raw-prompt-stdin", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_generator_bundle(config_path=args.config, adapter_path=args.adapter)

    if args.messages_stdin:
        messages = json.loads(sys.stdin.read())
        result = render_messages(
            bundle=bundle,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.messages_json:
        messages = json.loads(args.messages_json)
        result = render_messages(
            bundle=bundle,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.raw_prompt_stdin:
        result = render_raw_prompt(
            bundle=bundle,
            raw_prompt=sys.stdin.read(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.raw_prompt is not None:
        result = render_raw_prompt(
            bundle=bundle,
            raw_prompt=args.raw_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.prompt is not None:
        messages = []
        if args.system_prompt.strip():
            messages.append({"role": "system", "content": args.system_prompt.strip()})
        messages.append({"role": "user", "content": args.prompt.strip()})
        result = render_messages(
            bundle=bundle,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        raise SystemExit("Provide one of --prompt, --raw-prompt, --messages-json, --messages-stdin, or --raw-prompt-stdin.")

    if args.json:
        print(
            json.dumps(
                {
                    "prompt_text": result.prompt_text,
                    "completion": result.completion,
                    "full_text": result.full_text,
                },
                ensure_ascii=False,
            )
        )
        return

    print(result.completion)


if __name__ == "__main__":
    main()
