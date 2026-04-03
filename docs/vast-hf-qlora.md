# Vast.ai for HF QLoRA

Этот сценарий рассчитан на первый запуск:

- модель `Qwen/Qwen2.5-1.5B-Instruct`;
- `QLoRA`;
- одна GPU с `24 GB VRAM`;
- локально уже подготовленный датасет.

## Рабочее допущение

Для этого проекта считаем, что управление `vast.ai` выполняется прямо из локального окружения через `vastai` CLI:

- `vastai` установлен и доступен в `PATH`;
- аутентификация уже настроена;
- SSH-ключ для `vast.ai` уже подготовлен и используется для `vastai copy` и прямого `ssh`.

## Что уже подготовлено локально

После запуска `tinyllm-hf-prepare` артефакты лежат здесь:

- `data/processed/qwen25_1_5b_instruct_qlora_v1`
- `artifacts/qwen25_1_5b_instruct_qlora_v1`

Размер подготовленного датасета сейчас около `439 MB`, поэтому копировать его на Vast вполне нормально.

## Какой оффер искать

Ищем один GPU с `24 GB` и хорошей надежностью:

```bash
vastai search offers 'gpu_ram>=24 num_gpus=1 reliability>0.98 dph_total<0.35 inet_up>100 inet_down>100'
```

Предпочтительный класс карт:

- `RTX 3090 24 GB`
- `RTX A5000 24 GB`
- `A10 24 GB`

## Поднять инстанс

Можно использовать уже существующий helper:

```bash
scripts/vast-create-instance.sh OFFER_ID
```

Если хотите больший диск:

```bash
VAST_DISK_GB=60 scripts/vast-create-instance.sh OFFER_ID
```

`60 GB` здесь просто комфортный размер. Для этого запуска он с запасом.

## Проверенный рецепт запуска

После нескольких неудачных попыток рабочей оказалась максимально простая схема:

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 45 \
  --ssh \
  --direct \
  --cancel-unavail
```

Это важнее, чем desktop templates, Jupyter-образы или кастомный Docker image.

## Что скопировать на сервер

После создания инстанса:

```bash
vastai copy -i ~/.ssh/vast_ed25519 \
  data/processed/qwen25_1_5b_instruct_qlora_v1 \
  C.INSTANCE_ID:/app/data/processed/
```

Если репозиторий на инстансе уже не совпадает с локальным, проще скопировать весь проект:

```bash
vastai copy -i ~/.ssh/vast_ed25519 \
  . \
  C.INSTANCE_ID:/app
```

Но для экономии времени обычно достаточно:

- актуального кода репозитория;
- `data/processed/qwen25_1_5b_instruct_qlora_v1`.

## Запуск train

На сервере:

```bash
cd /app
uv run tinyllm-hf-train --config configs/qwen25_1_5b_instruct_qlora.toml
```

Результат появится в:

- `artifacts/qwen25_1_5b_instruct_qlora_v1/checkpoints`
- `artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`

## Как забрать адаптер обратно

```bash
vastai copy -i ~/.ssh/vast_ed25519 \
  C.INSTANCE_ID:/app/artifacts/qwen25_1_5b_instruct_qlora_v1 \
  ./artifacts/
```

## Рекомендуемый финальный цикл

Инстанс считается завершенным не в момент окончания train, а только после полного post-train цикла:

1. Подняли инстанс и дождались рабочего SSH.
2. Запустили обучение.
3. После завершения сделали быстрый smoke-test прямо на машине.
4. Скачали run целиком локально.
5. Сразу уничтожили инстанс.

Под это добавлен helper:

```bash
scripts/vast-hf-qlora-finalize.sh INSTANCE_ID
```

Что он делает:

- прогоняет `tinyllm-hf-smoke` на удаленной машине по нескольким фиксированным prompt'ам;
- сохраняет ответы в `artifacts/<run_name>/smoke/posttrain_smoke.jsonl`;
- скачивает `artifacts/<run_name>` и train-log в локальный `backups/vast/<timestamp>-<run_name>/`;
- локально строит финальный график обучения через `tinyllm-hf-plot`;
- вызывает `vastai destroy instance`.

Если нужно только скачать артефакты без destroy:

```bash
VAST_DESTROY=false scripts/vast-hf-qlora-finalize.sh INSTANCE_ID
```

Почему так:

- smoke-test нужно делать до destroy, пока адаптер еще лежит на машине и доступен без лишних копирований;
- plotting и прочую постобработку уже разумнее делать локально, без сжигания GPU-времени;
- destroy нужно делать сразу после успешной выгрузки, иначе инстанс просто продолжает стоить денег.

## Если словили OOM

Первый безопасный откат:

- уменьшить `max_length` с `1024` до `768` в `configs/qwen25_1_5b_instruct_qlora.toml`

Второй:

- уменьшить `gradient_accumulation_steps` не нужно;
- лучше оставить effective batch и снижать только длину контекста.

## Грабли и обходы

Что уже успели проверить на практике:

- Многие инстансы через Vast падали еще на этапе provisioning с ошибкой `No such container: C.<instance_id>`.
- `Linux Desktop Container` один раз действительно стартовал, но там не работали ни Jupyter/Portal, ни нормальный SSH-вход по ключу.
- Для этого пайплайна не нужен desktop/runtime path. Надежнее обычный `ssh`-запуск на `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
- На сервер лучше копировать только минимум: `pyproject.toml`, `uv.lock`, `configs/`, `src/` и `data/processed/qwen25_1_5b_instruct_qlora_v1`.
- При копировании надо следить за layout: подготовленный датасет должен лежать именно в `data/processed/qwen25_1_5b_instruct_qlora_v1`, иначе train падает с `Processed dataset not found`.
- На инстансе `uv` отсутствует по умолчанию. Его нужно ставить отдельно перед `uv sync --frozen`.
- В `trl==1.0.0` текущий `SFTConfig` не принимает аргумент `save_safetensors`. Это уже исправлено в локальном [hf_sft.py](/home/angel/projects/TinyLLM/src/tinyllm/hf_sft.py).
- Даже после правки train сначала продолжал брать старый установленный wheel. Рабочий обход: запускать не entrypoint, а модуль из актуального `src` через `PYTHONPATH=/workspace/TinyLLM/src`.

## Текущий живой run

По состоянию на `2026-04-02` baseline-тренировка уже идет:

- `instance_id`: `34031867`
- GPU: `RTX A5000 24 GB`
- tmux session: `qlora`
- лог: `/workspace/TinyLLM/logs/qwen25_1_5b_instruct_qlora_train.log`
- команда запуска:

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/workspace/TinyLLM/src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/TinyLLM
uv run python -m tinyllm.hf_sft --config configs/qwen25_1_5b_instruct_qlora.toml
```

Рабочие команды для проверки утром:

```bash
vastai show instance 34031867
vastai ssh-url 34031867
```

```bash
ssh -i ~/.ssh/vast_ed25519 -p PORT root@HOST \
  'tail -f /workspace/TinyLLM/logs/qwen25_1_5b_instruct_qlora_train.log'
```

```bash
ssh -i ~/.ssh/vast_ed25519 -p PORT root@HOST \
  'watch -n 2 nvidia-smi'
```
