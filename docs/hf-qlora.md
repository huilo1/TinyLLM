# HF QLoRA Pipeline

После учебной tiny-ветки следующий практический шаг в этом репозитории:

- взять готовую instruct/base модель;
- использовать ее родной tokenizer и chat template;
- дообучать через `QLoRA`, а не через full finetune.

Стартовый профиль здесь рассчитан на:

- модель `Qwen/Qwen2.5-1.5B-Instruct`;
- одну GPU с `24 GB VRAM`;
- `max_length = 1024`;
- `per_device_train_batch_size = 1`;
- `gradient_accumulation_steps = 16`.

Это консервативный первый запуск. Он нацелен на надежный старт без OOM, а не на максимум скорости.

Запуск именно на `vast.ai` для этого пайплайна описан в [docs/vast-hf-qlora.md](vast-hf-qlora.md).

## Зависимости

```bash
uv sync
```

Новый пайплайн использует:

- `transformers`
- `trl`
- `peft`
- `accelerate`
- `bitsandbytes`

## Шаг 1. Подготовить датасет

```bash
uv run tinyllm-hf-prepare --config configs/qwen25_1_5b_instruct_qlora.toml
```

Что делает шаг:

- скачивает исходный conversational dataset;
- приводит его к формату `messages`;
- ограничивает экстремально длинные диалоги;
- сохраняет подготовленный датасет на диск.

Подрезка длинных примеров сделана намеренно: для стартового `QLoRA`-run стабильность важнее, чем агрессивное покрытие хвоста распределения.

## Шаг 2. Запустить SFT

```bash
uv run tinyllm-hf-train --config configs/qwen25_1_5b_instruct_qlora.toml
```

Результаты будут сохраняться в:

- `artifacts/qwen25_1_5b_instruct_qlora_v1/checkpoints`
- `artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`

В `adapter/` лежит LoRA-адаптер, а не полный merged checkpoint.

## Шаг 3. Проверить генерацию

```bash
uv run tinyllm-hf-generate \
  --config configs/qwen25_1_5b_instruct_qlora.toml \
  --adapter artifacts/qwen25_1_5b_instruct_qlora_v1/adapter \
  --prompt "Объясни простыми словами, что такое attention."
```

## Шаг 4. Построить график обучения

После того как на диске появился хотя бы один `checkpoint-*`, можно построить аккуратный график train/eval-метрик из `trainer_state.json`:

```bash
uv run tinyllm-hf-plot --config configs/qwen25_1_5b_instruct_qlora.toml
```

Если вы сначала скачали run с Vast в локальный backup:

```bash
uv run tinyllm-hf-plot \
  --run-dir backups/vast/20260403-010357/qwen25_1_5b_instruct_qlora_v1
```

По умолчанию артефакты сохраняются в `<run_dir>/plots/`:

- `training_history.png`
- `training_history.svg`
- `train_metrics.csv`
- `eval_metrics.csv`
- `plot_summary.json`

Практически важно, что график строится по `trainer_state.json` из последнего checkpoint. Поэтому даже если `save_total_limit` удаляет старые checkpoint'ы, полная история метрик все равно остается доступной в самом свежем `trainer_state.json`.

## Когда нужен более мощный GPU

Имеет смысл уходить с `24 GB` на `32-48 GB`, если вы хотите:

- увеличить `max_length` до `2048+`;
- поднять `per_device_train_batch_size`;
- включить packing уже на первом запуске;
- обучать `base`-модель на continued pretraining перед SFT.

Для первого `QLoRA`-baseline это не требуется.
