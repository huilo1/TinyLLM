from __future__ import annotations

import argparse

import gradio as gr

from tinyllm.inference import build_news_prompt, load_generator


def build_demo(config_path: str, checkpoint_path: str, device: str | None):
    generator = load_generator(config_path, checkpoint_path, device=device)

    def run_structured(
        source: str,
        date: str,
        title: str,
        body_prefix: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[str, str, str]:
        prompt = build_news_prompt(
            source=source,
            date=date,
            title=title,
            body_prefix=body_prefix,
        )
        completion, full_text = generator.complete(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return prompt, completion, full_text

    def run_raw(
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[str, str]:
        completion, full_text = generator.complete(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return completion, full_text

    default_cfg = generator.config.generation
    with gr.Blocks(title="TinyLLM Russian Finance Demo") as demo:
        gr.Markdown(
            """
            # TinyLLM Russian Finance Demo

            Эта модель не является чат-моделью. Лучшие результаты она дает на промптах,
            похожих на обучающий формат новостей:
            `Источник -> Дата -> Заголовок -> Текст`.
            """
        )

        with gr.Tab("Структурный Промпт"):
            with gr.Row():
                source = gr.Textbox(label="Источник", value="finam")
                date = gr.Textbox(label="Дата", value="2024-01-15")
            title = gr.Textbox(
                label="Заголовок",
                value="Сбербанк увеличил чистую прибыль по РСБУ",
                lines=2,
            )
            body_prefix = gr.Textbox(
                label="Начало текста",
                placeholder="Необязательно. Можно задать первые слова статьи.",
                lines=4,
            )
            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=32,
                    maximum=256,
                    step=8,
                    value=default_cfg.max_new_tokens,
                    label="max_new_tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    step=0.1,
                    value=default_cfg.temperature,
                    label="temperature",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_cfg.top_k,
                    label="top_k (0 = без top-k)",
                )
            generate_structured = gr.Button("Сгенерировать")
            structured_prompt = gr.Textbox(label="Собранный prompt", lines=8)
            structured_completion = gr.Textbox(label="Только продолжение", lines=8)
            structured_full = gr.Textbox(label="Полный текст модели", lines=12)
            generate_structured.click(
                fn=run_structured,
                inputs=[
                    source,
                    date,
                    title,
                    body_prefix,
                    max_new_tokens,
                    temperature,
                    top_k,
                ],
                outputs=[structured_prompt, structured_completion, structured_full],
            )

        with gr.Tab("Raw Prompt"):
            raw_prompt = gr.Textbox(
                label="Raw prompt",
                value="Источник: finam\nДата: 2024-01-15\nЗаголовок:",
                lines=10,
            )
            with gr.Row():
                raw_max_new_tokens = gr.Slider(
                    minimum=32,
                    maximum=256,
                    step=8,
                    value=default_cfg.max_new_tokens,
                    label="max_new_tokens",
                )
                raw_temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    step=0.1,
                    value=default_cfg.temperature,
                    label="temperature",
                )
                raw_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_cfg.top_k,
                    label="top_k (0 = без top-k)",
                )
            generate_raw = gr.Button("Сгенерировать")
            raw_completion = gr.Textbox(label="Только продолжение", lines=10)
            raw_full = gr.Textbox(label="Полный текст модели", lines=14)
            generate_raw.click(
                fn=run_raw,
                inputs=[raw_prompt, raw_max_new_tokens, raw_temperature, raw_top_k],
                outputs=[raw_completion, raw_full],
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio demo for TinyLLM.")
    parser.add_argument("--config", default="configs/tiny_rfn.toml")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/tiny_rfn/checkpoints/best.pt",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
