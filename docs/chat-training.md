# TinyLLM Chat on ZeroAgency

Второй учебный шаг отличается от news-модели не только датасетом, но и целевой задачей:

- раньше модель училась продолжать новостной текст;
- теперь модель должна отвечать в формате диалога;
- значит, нужно менять и препроцессинг, и шаблон инференса, и сам train-loss.

## Что именно мы делаем

### 1. Проектируем локально

Локально дешевле и безопаснее сделать все, что не требует GPU:

- проверить формат датасета;
- собрать train/validation/test;
- обучить tokenizer;
- прогнать smoke-run;
- убедиться, что чатовый prompt и demo работают.

Почему это важно: ошибка в шаблоне чата или в токенизации на CPU стоит минуты, а на Vast это уже оплачиваемое GPU-время.

### 2. Учим на Vast

Основное обучение переносим на Vast, потому что даже маленькая чат-модель с контекстом `512` токенов заметно выигрывает от GPU.

Рекомендуемый базовый run:

- конфиг: `configs/tiny_chat.toml`
- датасет: `ZeroAgency/ru-instruct-conversation-v1`
- модель: около 29M параметров
- формат: causal LM с assistant-only loss

Assistant-only loss нужен потому, что для чата нас интересует именно качество ответов ассистента, а не воспроизведение пользовательских реплик как обучающей цели.

### 3. Инференс

Для локального qualitative-check GPU не обязателен:

- этот проект обучает маленькую модель с нуля, а не многомиллиардный LLM;
- значит, локальный CPU-инференс возможен;
- но отвечать он будет медленнее, чем на GPU.

Практический вывод:

- smoke и ручные проверки можно делать локально на CPU;
- массовый интерактивный inference лучше держать на GPU;
- для учебной цели локального CPU после обучения достаточно.

## Пошаговый план

### Шаг 1. Подготовить chat-датасет

```bash
uv run tinyllm-prepare --config configs/tiny_chat.toml
```

Что происходит:

- датасет скачивается с Hugging Face;
- сообщения приводятся к chat-шаблону `<|system|> / <|user|> / <|assistant|>`;
- если у датасета нет `test`, он выделяется автоматически;
- сохраняются `text` для tokenizer и `messages` для masked training.

### Шаг 2. Обучить tokenizer

```bash
uv run tinyllm-train-tokenizer --config configs/tiny_chat.toml
```

Почему отдельно:

- tokenizer должен видеть только train split;
- чатовый шаблон с role markers тоже должен попасть в словарь;
- это делает генерацию формата более стабильной.

### Шаг 3. Прогнать локальный smoke-run

```bash
uv run tinyllm-prepare --config configs/smoke_chat.toml
uv run tinyllm-train-tokenizer --config configs/smoke_chat.toml
uv run tinyllm-cache-tokens --config configs/smoke_chat.toml --splits train validation
uv run tinyllm-train --config configs/smoke_chat.toml
```

Что проверяем:

- пайплайн не падает;
- chat loss считается корректно;
- checkpoint сохраняется;
- sample generations выходят в чат-формате.

### Шаг 4. Материализовать токены перед GPU-run

Локально:

```bash
uv run tinyllm-cache-tokens --config configs/tiny_chat.toml --splits train validation
```

Это уменьшает оплачиваемое время на GPU: Vast тратит часы именно на train, а не на токенизацию.

### Шаг 5. Обучить основную модель на Vast

В контейнере на Vast:

```bash
uv run tinyllm-train --config configs/tiny_chat.toml
```

Минимально разумный класс GPU для старта:

- `RTX 3090 / 4090 / A10 / A5000` и выше

Если памяти не хватает:

- сначала уменьшать `batch_size`;
- только потом сокращать `sequence_length`.

### Шаг 6. Оценить и проверить ответы

```bash
uv run tinyllm-eval --config configs/tiny_chat.toml --checkpoint artifacts/tiny_chat_v1/checkpoints/best.pt --split test
uv run tinyllm-generate --config configs/tiny_chat.toml --checkpoint artifacts/tiny_chat_v1/checkpoints/best.pt --prompt "<|user|>\nОбъясни, что такое attention.\n\n<|assistant|>"
```

### Шаг 7. Поднять локальный demo UI

```bash
uv run tinyllm-demo --config configs/tiny_chat.toml --checkpoint artifacts/tiny_chat_v1/checkpoints/best.pt --device cpu
```

Для локальной машины без GPU это нормальный режим. Если позже появится CUDA, можно заменить `--device cpu` на `--device cuda`.
