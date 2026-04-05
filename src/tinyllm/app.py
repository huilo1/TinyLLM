from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gradio as gr

from tinyllm.inference import build_news_prompt, load_generator
from tinyllm.remote_inference import HFRemoteSSHGenerator, RemoteInferenceError, RemoteWorkerUnavailable

WEB_MAX_NEW_TOKENS_LIMIT = 240


def _raise_ui_error(exc: Exception) -> None:
    raise gr.Error(str(exc)) from exc


def resolve_training_plot_path(run_dir: Path | None, checkpoint_path: str | Path | None, explicit_path: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    if run_dir is not None:
        candidates.append(run_dir / "plots" / "training_history.png")
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        candidates.extend(
            [
                checkpoint.parent.parent / "plots" / "training_history.png",
                checkpoint.parent / "plots" / "training_history.png",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized
    return None


def load_report_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_smoke_rows(path: Path, limit: int = 3) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def render_runtime_controls(generator) -> None:
    unload_fn = getattr(generator, "unload", None)
    if not callable(unload_fn):
        return

    def unload_remote_model() -> str:
        try:
            return unload_fn()
        except (RemoteWorkerUnavailable, RemoteInferenceError) as exc:
            _raise_ui_error(exc)

    with gr.Row():
        unload_button = gr.Button("Выгрузить модель", variant="stop")
        unload_status = gr.Textbox(
            label="Состояние GPU worker",
            value="Модель поднимается автоматически по первому запросу и после reboot GPU-хоста.",
            interactive=False,
        )
    unload_button.click(fn=unload_remote_model, inputs=[], outputs=[unload_status])


def render_training_report_section(report_dir: Path | None, training_plot_path: Path | None) -> None:
    plot_summary = load_report_json(report_dir / "plots" / "plot_summary.json") if report_dir else None
    train_setup = load_report_json(report_dir / "hf_train_setup.json") if report_dir else None
    if train_setup is None and report_dir is not None:
        train_setup = load_report_json(report_dir / "train_setup.json")
    smoke_rows = load_smoke_rows(report_dir / "smoke" / "posttrain_smoke.jsonl") if report_dir else []

    with gr.Accordion("Обучение И Метрики", open=True):
        if plot_summary:
            latest_step = plot_summary.get("latest_step")
            latest_eval_loss = plot_summary.get("latest_eval_loss")
            latest_eval_accuracy = plot_summary.get("latest_eval_accuracy")
            eval_points = plot_summary.get("eval_points")
            train_points = plot_summary.get("train_points")
            gr.Markdown(
                "\n".join(
                    [
                        "### Итог По Обучению",
                        f"- `latest_step`: `{latest_step}`",
                        f"- `latest_eval_loss`: `{latest_eval_loss:.4f}`" if isinstance(latest_eval_loss, (int, float)) else f"- `latest_eval_loss`: `{latest_eval_loss}`",
                        f"- `latest_eval_accuracy`: `{latest_eval_accuracy:.4f}`" if isinstance(latest_eval_accuracy, (int, float)) else f"- `latest_eval_accuracy`: `{latest_eval_accuracy}`",
                        f"- `train_points`: `{train_points}`",
                        f"- `eval_points`: `{eval_points}`",
                    ]
                )
            )

        if training_plot_path is None:
            gr.Markdown(
                "График появится автоматически, когда рядом с report-артефактами будет доступен `plots/training_history.png`."
            )
        else:
            gr.Image(
                value=str(training_plot_path),
                label="Train / Eval Dynamics",
                interactive=False,
            )

        if train_setup:
            with gr.Accordion("Параметры Обучения", open=False):
                gr.JSON(value=train_setup, label="Train setup")

        if smoke_rows:
            with gr.Accordion("Post-Train Smoke", open=False):
                for row in smoke_rows:
                    gr.Markdown(
                        "\n".join(
                            [
                                f"**{row.get('id', 'sample')}**",
                                f"`prompt`: {row.get('prompt', '')}",
                                f"`watch`: {row.get('watch', '')}",
                                f"`response`: {row.get('response', '')}",
                            ]
                        )
                    )


def build_chat_demo(generator, report_dir: Path | None = None, training_plot_path: Path | None = None):
    def run_chat(
        message: str,
        history: list[dict] | None,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[list[dict], str, str, str]:
        history = history or []
        prompt_history = [
            item
            for item in history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and str(item.get("content", "")).strip()
        ]

        try:
            reply, prompt, full_text = generator.chat(
                user_message=message,
                system_prompt=system_prompt,
                history=prompt_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except (RemoteWorkerUnavailable, RemoteInferenceError) as exc:
            _raise_ui_error(exc)
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return updated_history, prompt, full_text, ""

    def run_raw(
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[str, str]:
        try:
            completion, full_text = generator.complete(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except (RemoteWorkerUnavailable, RemoteInferenceError) as exc:
            _raise_ui_error(exc)
        return completion, full_text

    default_cfg = generator.config.generation
    default_max_new_tokens = min(default_cfg.max_new_tokens, WEB_MAX_NEW_TOKENS_LIMIT)
    with gr.Blocks(title="TinyLLM Russian Chat Demo") as demo:
        gr.Markdown(
            """
            # TinyLLM Russian Chat Demo

            Это instruction-tuned QLoRA-адаптер на базе `Qwen2.5-1.5B-Instruct`.
            Веб-интерфейс работает на прод-сервере, а генерация идет на удаленной GPU-машине через SSH-backed worker.
            """
        )
        render_runtime_controls(generator)

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
                    maximum=WEB_MAX_NEW_TOKENS_LIMIT,
                    step=8,
                    value=default_max_new_tokens,
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
                    maximum=WEB_MAX_NEW_TOKENS_LIMIT,
                    step=8,
                    value=default_max_new_tokens,
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

        render_training_report_section(report_dir, training_plot_path)

    return demo


def build_news_demo(generator, report_dir: Path | None = None, training_plot_path: Path | None = None):
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
        try:
            completion, full_text = generator.complete(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except (RemoteWorkerUnavailable, RemoteInferenceError) as exc:
            _raise_ui_error(exc)
        return prompt, completion, full_text

    def run_raw(
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[str, str]:
        try:
            completion, full_text = generator.complete(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except (RemoteWorkerUnavailable, RemoteInferenceError) as exc:
            _raise_ui_error(exc)
        return completion, full_text

    default_cfg = generator.config.generation
    default_max_new_tokens = min(default_cfg.max_new_tokens, WEB_MAX_NEW_TOKENS_LIMIT)
    with gr.Blocks(title="TinyLLM Russian Finance Demo") as demo:
        gr.Markdown(
            """
            # TinyLLM Russian Finance Demo

            Это demo для inference через прод-вебморду. Лучшие результаты она дает на промптах,
            похожих на обучающий формат новостей:
            `Источник -> Дата -> Заголовок -> Текст`.
            """
        )
        render_runtime_controls(generator)

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
                    maximum=WEB_MAX_NEW_TOKENS_LIMIT,
                    step=8,
                    value=default_max_new_tokens,
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
                    maximum=WEB_MAX_NEW_TOKENS_LIMIT,
                    step=8,
                    value=default_max_new_tokens,
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

        render_training_report_section(report_dir, training_plot_path)

    return demo


def build_demo(
    backend: str,
    config_path: str,
    checkpoint_path: str | None,
    device: str | None,
    report_dir: str | None = None,
    training_plot_path: str | None = None,
    remote_ssh_host: str | None = None,
    remote_ssh_port: int = 2222,
    remote_ssh_user: str = "angel",
    remote_workdir: str | None = None,
    remote_activate: str | None = None,
    remote_config: str | None = None,
    remote_adapter: str | None = None,
    remote_max_new_tokens: int = 240,
    remote_temperature: float = 0.5,
    remote_top_k: int = 40,
):
    resolved_report_dir = Path(report_dir) if report_dir else None

    if backend == "remote-hf":
        if not all([remote_ssh_host, remote_workdir, remote_activate, remote_config, remote_adapter]):
            raise ValueError("Remote HF backend requires SSH host, workdir, remote activate path, remote config, and remote adapter path.")
        generator = HFRemoteSSHGenerator(
            ssh_host=remote_ssh_host,
            ssh_port=remote_ssh_port,
            ssh_user=remote_ssh_user,
            remote_workdir=remote_workdir,
            remote_activate_path=remote_activate,
            remote_config_path=remote_config,
            remote_adapter_path=remote_adapter,
            report_dir=resolved_report_dir or Path("artifacts/qwen25_1_5b_instruct_qlora_v1"),
            max_new_tokens=remote_max_new_tokens,
            temperature=remote_temperature,
            top_k=remote_top_k,
        )
    else:
        if checkpoint_path is None:
            raise ValueError("Local backend requires --checkpoint.")
        generator = load_generator(config_path, checkpoint_path, device=device)
        if resolved_report_dir is None:
            resolved_report_dir = generator.config.run_dir

    resolved_plot_path = resolve_training_plot_path(
        run_dir=resolved_report_dir or getattr(generator.config, "run_dir", None),
        checkpoint_path=checkpoint_path,
        explicit_path=training_plot_path,
    )
    if generator.config.is_chat_model:
        return build_chat_demo(generator, report_dir=resolved_report_dir, training_plot_path=resolved_plot_path)
    return build_news_demo(generator, report_dir=resolved_report_dir, training_plot_path=resolved_plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio demo for TinyLLM.")
    parser.add_argument("--backend", choices=["local", "remote-hf"], default=os.getenv("APP_BACKEND", "local"))
    parser.add_argument("--config", default=os.getenv("APP_CONFIG", "configs/tiny_chat_wide.toml"))
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("APP_CHECKPOINT", "artifacts/tiny_chat_wide_v1/checkpoints/best.pt"),
    )
    parser.add_argument("--report-dir", default=os.getenv("APP_REPORT_DIR"))
    parser.add_argument(
        "--training-plot",
        default=os.getenv("APP_TRAINING_PLOT"),
        help="Optional path to training_history.png. If omitted, the app will auto-discover it in run artifacts.",
    )
    parser.add_argument("--device", default=os.getenv("APP_DEVICE"), choices=["cpu", "cuda"])
    parser.add_argument("--host", default=os.getenv("APP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "7860")))
    parser.add_argument("--remote-ssh-host", default=os.getenv("REMOTE_SSH_HOST"))
    parser.add_argument("--remote-ssh-port", type=int, default=int(os.getenv("REMOTE_SSH_PORT", "2222")))
    parser.add_argument("--remote-ssh-user", default=os.getenv("REMOTE_SSH_USER", "angel"))
    parser.add_argument("--remote-workdir", default=os.getenv("REMOTE_WORKDIR"))
    parser.add_argument("--remote-activate", default=os.getenv("REMOTE_VENV_ACTIVATE"))
    parser.add_argument("--remote-config", default=os.getenv("REMOTE_CONFIG"))
    parser.add_argument("--remote-adapter", default=os.getenv("REMOTE_ADAPTER"))
    parser.add_argument("--remote-max-new-tokens", type=int, default=int(os.getenv("REMOTE_MAX_NEW_TOKENS", "240")))
    parser.add_argument("--remote-temperature", type=float, default=float(os.getenv("REMOTE_TEMPERATURE", "0.5")))
    parser.add_argument("--remote-top-k", type=int, default=int(os.getenv("REMOTE_TOP_K", "40")))
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(
        backend=args.backend,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        report_dir=args.report_dir,
        training_plot_path=args.training_plot,
        remote_ssh_host=args.remote_ssh_host,
        remote_ssh_port=args.remote_ssh_port,
        remote_ssh_user=args.remote_ssh_user,
        remote_workdir=args.remote_workdir,
        remote_activate=args.remote_activate,
        remote_config=args.remote_config,
        remote_adapter=args.remote_adapter,
        remote_max_new_tokens=args.remote_max_new_tokens,
        remote_temperature=args.remote_temperature,
        remote_top_k=args.remote_top_k,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
