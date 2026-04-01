# TinyLLM on RussianFinancialNews

Учебный проект: мы строим маленькую decoder-only language model с нуля на датасете [`Kasymkhan/RussianFinancialNews`](https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews).

Здесь принципиально важно:

1. Мы не используем pretrained weights.
2. Мы не используем готовый tokenizer.
3. Мы обучаем и tokenizer, и модель с нуля на одном корпусе.

Из-за этого качество будет ограниченным. Это ожидаемо: цель проекта не в сильной продакшн-модели, а в том, чтобы пройти весь цикл обучения LLM своими руками.

## Почему проект устроен именно так

### 1. Сначала локальная подготовка, потом GPU

`vast.ai` нужен только там, где без GPU действительно больно:

- основное обучение модели;
- при желании более длинный повторный запуск.

Все остальное дешевле и безопаснее сделать локально:

- скачать и нормализовать датасет;
- обучить tokenizer;
- проверить shapes, loss и генерацию на коротком smoke-run;
- подготовить конфиги и checkpointing.

Это экономит деньги: ошибка в препроцессинге на локальной машине стоит минуты CPU, а не оплачиваемое время аренды GPU.

### 2. Почему decoder-only Transformer

Это самый прямой способ понять, как работает LLM:

- входной текст разбивается на токены;
- модель предсказывает следующий токен;
- обучение идет по causal language modeling loss.

Мы сознательно не берем encoder-decoder архитектуру и не делаем классификатор, потому что цель именно tiny LLM.

### 3. Почему свой tokenizer

Если взять чужой tokenizer, часть “магии” уже будет унаследована от чужой системы. Для учебной цели честнее обучить свой byte-level BPE tokenizer на нашем корпусе. Он:

- работает с русским текстом без ручной таблицы символов;
- не ломается на редких символах;
- остается достаточно простым для изучения.

### 4. Почему модель маленькая

Датасет небольшой: около 92 тысяч новостей. Для такого корпуса большая модель не имеет смысла:

- она будет дольше и дороже обучаться;
- будет сильнее переобучаться;
- учебная польза не вырастет пропорционально.

Поэтому базовый конфиг здесь нацелен на модель порядка десятков миллионов параметров, а не сотен миллионов.

## Структура проекта

- `configs/tiny_rfn.toml` — основной конфиг.
- `configs/smoke.toml` — очень маленький запуск для локальной проверки.
- `src/tinyllm/prepare_dataset.py` — загрузка и нормализация датасета.
- `src/tinyllm/tokenizer.py` — обучение tokenizer с нуля.
- `src/tinyllm/model.py` — tiny GPT-подобная модель на PyTorch.
- `src/tinyllm/train.py` — обучение.
- `src/tinyllm/evaluate.py` — оценка loss / perplexity.
- `src/tinyllm/generate.py` — генерация текста.
- `src/tinyllm/app.py` — локальный Gradio UI для тестирования модели.

## Пайплайн

### Шаг 1. Установить зависимости

```bash
uv venv
uv pip install -e .
```

Почему так: `uv` уже доступен локально и хорошо подходит для быстрого, воспроизводимого окружения.

### Шаг 2. Подготовить датасет

```bash
uv run tinyllm-prepare --config configs/tiny_rfn.toml
```

Что делает шаг:

- скачивает `Kasymkhan/RussianFinancialNews`;
- делает `train/validation/test`;
- нормализует текст;
- сохраняет итоговый `DatasetDict` на диск.

Нормализация намеренно простая:

- `source` и `date` сохраняем как контекст;
- `title` добавляем только если он осмысленный;
- `body` считаем главным текстом;
- слишком короткие записи отбрасываем как шум.

### Шаг 3. Обучить tokenizer

```bash
uv run tinyllm-train-tokenizer --config configs/tiny_rfn.toml
```

Что делает шаг:

- читает только `train` split;
- учит BPE tokenizer с нуля;
- сохраняет `tokenizer.json` и метаданные.

Использовать только `train` важно, чтобы tokenizer не видел тестовые тексты.

### Шаг 4. Локальный smoke run

```bash
uv run tinyllm-prepare --config configs/smoke.toml
uv run tinyllm-train-tokenizer --config configs/smoke.toml
uv run tinyllm-cache-tokens --config configs/smoke.toml
uv run tinyllm-train --config configs/smoke.toml
```

Цель этого запуска:

- проверить, что код работает;
- убедиться, что loss считается;
- убедиться, что checkpoint сохраняется;
- не платить за GPU до тех пор, пока pipeline не валидирован.

### Шаг 5. Основное обучение

Локально на CPU можно запускать только smoke-версию. Основной train лучше переносить на `vast.ai`.

Перед GPU-запуском стоит заранее материализовать token cache локально:

```bash
uv run tinyllm-cache-tokens --config configs/tiny_rfn.toml --splits train validation
```

Почему это важно: токенизация корпуса не требует GPU, но может занять заметное время. Если сделать этот шаг заранее, на `vast.ai` оплачиваемое время уйдет именно на обучение, а не на подготовку последовательностей.

```bash
uv run tinyllm-train --config configs/tiny_rfn.toml
```

### Шаг 6. Оценка и генерация

```bash
uv run tinyllm-eval --config configs/tiny_rfn.toml --checkpoint artifacts/tiny_rfn/checkpoints/best.pt
uv run tinyllm-generate --config configs/tiny_rfn.toml --checkpoint artifacts/tiny_rfn/checkpoints/best.pt --prompt "Источник: finam\nДата: 2024-01-15\nЗаголовок:"
```

### Шаг 7. Локальный веб-интерфейс

```bash
uv run tinyllm-demo --config configs/tiny_rfn.toml --checkpoint artifacts/tiny_rfn/checkpoints/best.pt --device cpu
```

После запуска откройте:

```text
http://127.0.0.1:7860
```

Почему интерфейс не сделан как чат:

- модель обучалась не на диалогах, а на финансовых новостях;
- поэтому лучший режим тестирования для нее это completion по news-like prompt;
- в UI есть два режима: структурный промпт и raw prompt.

Если хотите принудительно использовать GPU локально, можно заменить `--device cpu` на `--device cuda`.

## GitHub и деплой

Код проекта лежит в GitHub-репозитории:

```text
https://github.com/huilo1/TinyLLM
```

Tokenizer и маленькие inference-метаданные уже лежат в репозитории:

- `artifacts/tiny_rfn/tokenizer/tokenizer.json`
- `artifacts/tiny_rfn/tokenizer/tokenizer_meta.json`
- `artifacts/tiny_rfn/train_setup.json`
- `artifacts/tiny_rfn/eval_test.json`

Большой checkpoint не хранится в обычном git, потому что GitHub ограничивает размер файлов. Он лежит в Release:

```text
https://github.com/huilo1/TinyLLM/releases/tag/model-v1
```

Прямая ссылка на `best.pt`:

```text
https://github.com/huilo1/TinyLLM/releases/download/model-v1/best.pt
```

Для деплоя на хостинге это значит:

1. Хостинг забирает код из репозитория.
2. `best.pt` скачивается отдельно из GitHub Release.
3. Приложение запускается с путем к этому checkpoint.

Чтобы деплой понял, что нужно тянуть новую модель, в репозитории есть файл:

```text
deploy/model-release.env
```

В нем хранятся:

- `MODEL_VERSION`
- `CHECKPOINT_URL`
- `CHECKPOINT_SHA256`

Практический процесс обновления модели такой:

1. Загружаете новый `best.pt` в GitHub Release.
2. Обновляете `deploy/model-release.env` под новый release и новый `sha256`.
3. Делаете `git push` в `main`.

После этого автодеплой:

1. подтянет свежий код;
2. увидит, что манифест модели изменился;
3. скачает новый checkpoint;
4. проверит `sha256`;
5. перезапустит приложение уже на новой модели.

## Когда подключать vast.ai

Подключать GPU стоит только после того, как локально пройдены:

1. `prepare_dataset`
2. `train_tokenizer`
3. `train` на `configs/smoke.toml`

Это критично для экономии денег.

Рекомендуемый порядок на `vast.ai`:

1. Выбрать один GPU с `24 GB VRAM` уровня `RTX 3090 / 4090 / A5000`.
2. Поднять инстанс только на время обучения.
3. Скопировать проект.
4. Запустить сначала короткий GPU smoke run.
5. Если все стабильно, запустить основной train.
6. Скопировать артефакты обратно.
7. Сразу уничтожить инстанс.

Подробнее: [docs/vastai.md](docs/vastai.md)

## Важное ограничение

Эта модель обучается на маленьком, доменно-узком корпусе. Поэтому стоит ожидать:

- слабую общую языковую компетентность;
- повторение шаблонов финансовых новостей;
- чувствительность к формату промпта;
- ограниченную длину связного текста.

Это нормально. Для учебного проекта это даже полезно: так легче увидеть реальные ограничения маленьких моделей.
