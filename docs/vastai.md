# Vast.ai workflow

Ниже зафиксирован workflow, который минимизирует оплачиваемое время GPU.

## Почему не арендуем сервер сразу

На `vast.ai` оплата идет по времени. Значит дорого делать на GPU то, что можно заранее проверить локально:

- загрузку датасета;
- подготовку split'ов;
- обучение tokenizer;
- корректность модели;
- сохранение checkpoint;
- короткую генерацию.

Правило проекта простое: пока локальный smoke-run не зеленый, GPU не арендуем.

## Что ставить на локальной машине заранее

Официальный CLI Vast.ai сейчас называется `vastai`.

Правильнее ставить его отдельно от ML-окружения проекта:

```bash
uv tool install vastai
```

Почему не `uv pip install vastai` в проектный `.venv`:

- `vastai` тянет свои инфраструктурные зависимости;
- они могут конфликтовать с версиями библиотек в ML-стеке;
- отдельная tool-установка сохраняет training environment чистым.

Дальше нужен API key, который обычно передается через `~/.vast_api_key` или `--api-key`.

## Рекомендуемый сценарий

### 1. Найти подходящие офферы

Сначала ищем один недорогой GPU на 24 GB VRAM:

```bash
vastai search offers 'gpu_ram>=24 num_gpus=1 reliability>0.98 dph_total<0.35 inet_up>100 inet_down>100'
```

Почему так:

- `gpu_ram>=24` нужен запас под модель, optimizer states и батчи;
- `num_gpus=1` дешевле и проще для первого учебного запуска;
- `reliability>0.98` снижает шанс внезапных проблем;
- лимит по цене удерживает проект в учебном бюджете.

### 2. Поднять инстанс только под обучение

После выбора `offer_id`:

```bash
vastai create instance OFFER_ID --image nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 --disk 40 --ssh --direct
```

Почему такой образ:

- обычный CUDA-образ проще для ручной установки Python-окружения;
- меньше скрытой магии, чем в “готовых ML images”.

### 3. На сервере

```bash
apt-get update && apt-get install -y python3.11 python3.11-venv git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-or-copy-project>
cd TinyLLM
uv venv
uv pip install -e .
```

Сначала запускаем короткую проверку:

```bash
uv run tinyllm-train --config configs/smoke.toml
```

И только потом основной train:

```bash
uv run tinyllm-train --config configs/tiny_rfn.toml
```

### 4. После окончания

Забираем папку `artifacts/.../checkpoints`, затем:

```bash
vastai destroy instance INSTANCE_ID
```

Почему именно `destroy`, а не просто `stop`:

- `destroy` гарантированно прекращает оплату инстанса;
- для учебного сценария нам обычно выгоднее сохранить артефакты локально и закрыть сервер полностью.

## Практика экономии

Перед запуском на GPU всегда проверьте:

1. Датасет сохранен локально без ошибок.
2. Tokenizer обучен и лежит в `artifacts/.../tokenizer`.
3. Token cache для `train` и `validation` уже создан локально.
4. Smoke-конфиг завершился локально.
5. Понятно, где будут сохраняться checkpoints.
6. Есть команда для копирования артефактов обратно.

Если один из этих пунктов не выполнен, аренда GPU преждевременна.
