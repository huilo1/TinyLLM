# Vast.ai with a reusable Docker image

Идея правильная: на `vast.ai` имеет смысл возить с собой не “сырую Ubuntu + ручной bootstrap”, а уже собранный image c Python, `uv`, PyTorch и кодом проекта.

Что это дает:

- меньше оплачиваемого времени на установку окружения;
- меньше расхождений между хостами;
- воспроизводимый train/inference runtime;
- проще отлаживать несовместимости CUDA и PyTorch один раз, а не на каждом инстансе.

## Что этот Dockerfile решает

`Dockerfile` в корне проекта:

- стартует от `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`;
- ставит `uv`;
- докладывает только project-specific зависимости;
- копирует код и оба tokenizer-пути: `tiny_rfn` и `tiny_rfn_v2`;
- оставляет контейнер универсальным для train, eval и demo.

Важно:

- образ не запекает в себя `data/processed/...`, token cache и checkpoints;
- эти большие артефакты лучше хранить отдельно и копировать только когда они реально нужны.

## Локальная сборка

```bash
docker build -t tinyllm:cu124 .
```

## Если захотите возить image через registry

Пример с `ghcr.io`:

```bash
docker tag tinyllm:cu124 ghcr.io/huilo1/tinyllm:cu124-v2
docker push ghcr.io/huilo1/tinyllm:cu124-v2
```

Сейчас этот образ уже опубликован:

```text
ghcr.io/huilo1/tinyllm:cu124-v2
```

Digest:

```text
sha256:621c73ea54447d573ddb34292d9ab7cf94aeacb151619a6eda3739fdd39f1363
```

## Запуск на Vast.ai

Да, `vastai create instance` принимает image напрямую через `--image`.

Пример:

```bash
vastai create instance OFFER_ID \
  --image ghcr.io/huilo1/tinyllm:cu124-v2 \
  --disk 40 \
  --ssh \
  --direct
```

И для этого же сценария добавлен маленький helper-скрипт:

```bash
scripts/vast-create-instance.sh OFFER_ID
```

По умолчанию он стартует `ssh`-инстанс с образом `ghcr.io/huilo1/tinyllm:cu124-v2`, но это можно переопределить через env:

```bash
VAST_IMAGE=ghcr.io/huilo1/tinyllm:cu124-v2 \
VAST_DISK_GB=50 \
scripts/vast-create-instance.sh OFFER_ID
```

После этого на сервере уже не нужно ставить Python и зависимости с нуля. Дальше остается только:

- скопировать `data/processed/tiny_rfn_v2`;
- скопировать `artifacts/tiny_rfn_v2/cache`, если хотим экономить GPU-время;
- запустить train командой из контейнера.

## SSH-ключ для Vast

Теперь выделенный ключ для Vast:

- private key: `~/.ssh/vast_ed25519`
- public key: `~/.ssh/vast_ed25519.pub`

Он добавлен в аккаунт Vast и правило для `ssh*.vast.ai` уже прописано в `~/.ssh/config`.

Для прямого доступа к инстансам по IP и нестандартному порту используйте:

```bash
ssh -i ~/.ssh/vast_ed25519 -o IdentitiesOnly=yes -p PORT root@HOST
```

Для `vastai copy` используйте тот же ключ явно:

```bash
vastai copy -i ~/.ssh/vast_ed25519 local:/path C.INSTANCE_ID:/workspace/path
```
