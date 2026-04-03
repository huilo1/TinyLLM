# Deploy Notes

Это короткая operational-памятка для прод-деплоя `https://tinyllm.runningdog.org/`.

Подробный runbook:

```text
DEPLOYMENT.md
```

Базовый deploy entrypoint:

```text
deploy/deploy-prod.sh
```

## Что происходит при push в main

1. GitHub Actions checkout'ит текущий commit.
2. Запускает `bash deploy/deploy-prod.sh`.
3. Скрипт читает `deploy/model-release.env`.
4. Скрипт делает preflight-checks.
5. Скрипт вызывает `/opt/apps/deploy-tinyllm.sh prod`.
6. Скрипт делает healthcheck URL.

## Где настраивается прод

Основные переменные лежат в:

```text
deploy/model-release.env
```

Ключевые поля:

- `APP_BACKEND=remote-hf`
- `APP_REPORT_DIR=artifacts/qwen25_1_5b_instruct_qlora_v1`
- `REMOTE_SSH_HOST=localhost`
- `REMOTE_SSH_PORT=2222`
- `REMOTE_SSH_USER=angel`
- `REMOTE_WORKDIR=/home/angel/projects/TinyLLM`
- `REMOTE_CONFIG=configs/qwen25_1_5b_instruct_qlora.toml`
- `REMOTE_ADAPTER=artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`

## Если деплой упал

Проверь по порядку:

1. Есть ли в репо report assets:
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/training_history.png`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/plot_summary.json`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/hf_train_setup.json`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/smoke/posttrain_smoke.jsonl`

2. Есть ли SSH-доступ с deploy-сервера:

```bash
ssh -p 2222 angel@localhost 'echo ok'
```

3. Есть ли на GPU-машине нужные пути:

```bash
ssh -p 2222 angel@localhost \
  'test -d /home/angel/projects/TinyLLM && \
   test -f /home/angel/projects/TinyLLM/configs/qwen25_1_5b_instruct_qlora.toml && \
   test -d /home/angel/projects/TinyLLM/artifacts/qwen25_1_5b_instruct_qlora_v1/adapter'
```

4. Есть ли platform wrapper:

```bash
test -x /opt/apps/deploy-tinyllm.sh
```

5. Открывается ли healthcheck URL после выкладки:

```bash
curl -I https://tinyllm.runningdog.org/
```

## Что обновлять при новой модели

1. Обновить remote target в `deploy/model-release.env`, если сменились пути.
2. Положить в git свежие report assets.
3. Убедиться, что adapter уже лежит на GPU-машине.
4. Сделать push в `main`.
