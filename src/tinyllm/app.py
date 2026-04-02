from __future__ import annotations

import argparse

import gradio as gr

from tinyllm.inference import build_news_prompt, load_generator


def build_chat_demo(generator):
    def run_chat(
        message: str,
        history: list[tuple[str, str]] | None,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[list[tuple[str, str]], str, str, str]:
        history = history or []
        prompt_history = []
        for user_text, assistant_text in history:
            if user_text:
                prompt_history.append({"role": "user", "content": user_text})
            if assistant_text:
                prompt_history.append({"role": "assistant", "content": assistant_text})

        reply, prompt, full_text = generator.chat(
            user_message=message,
            system_prompt=system_prompt,
            history=prompt_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        updated_history = history + [(message, reply)]
        return updated_history, prompt, full_text, ""

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
    with gr.Blocks(title="TinyLLM Russian Chat Demo") as demo:
        gr.Markdown(
            """
            # TinyLLM Russian Chat Demo

            Это маленькая чат-модель, обученная с нуля. Для локальной qualitative-проверки CPU достаточно,
            но отвечать она будет медленнее, чем на GPU.
            """
        )

        with gr.Tab("Чат"):
            system_prompt = gr.Textbox(
                label="System prompt",
                value="Ты полезный русскоязычный ассистент. Отвечай ясно и по делу.",
                lines=3,
            )
            chatbot = gr.Chatbot(label="Диалог", height=420)
            user_message = gr.Textbox(label="Сообщение", lines=4, placeholder="Напишите вопрос...")
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
            with gr.Row():
                send = gr.Button("Отправить", variant="primary")
                clear = gr.Button("Очистить диалог")
            assembled_prompt = gr.Textbox(label="Собранный prompt", lines=12)
            full_output = gr.Textbox(label="Полный текст модели", lines=12)

            send.click(
                fn=run_chat,
                inputs=[user_message, chatbot, system_prompt, max_new_tokens, temperature, top_k],
                outputs=[chatbot, assembled_prompt, full_output, user_message],
            )
            user_message.submit(
                fn=run_chat,
                inputs=[user_message, chatbot, system_prompt, max_new_tokens, temperature, top_k],
                outputs=[chatbot, assembled_prompt, full_output, user_message],
            )
            clear.click(
                fn=lambda: ([], "", "", ""),
                inputs=[],
                outputs=[chatbot, assembled_prompt, full_output, user_message],
            )

        with gr.Tab("Raw Prompt"):
            raw_prompt = gr.Textbox(
                label="Raw prompt",
                value="<|system|>\nТы полезный русскоязычный ассистент.\n\n<|user|>\nОбъясни, что такое LoRA.\n\n<|assistant|>",
                lines=12,
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


def build_news_demo(generator):
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
                inputs=[source, date, title, body_prefix, max_new_tokens, temperature, top_k],
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


def build_demo(config_path: str, checkpoint_path: str, device: str | None):
    generator = load_generator(config_path, checkpoint_path, device=device)
    if generator.config.is_chat_model:
        return build_chat_demo(generator)
    return build_news_demo(generator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio demo for TinyLLM.")
    parser.add_argument("--config", default="configs/tiny_chat_wide.toml")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/tiny_chat_wide_v1/checkpoints/best.pt",
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
