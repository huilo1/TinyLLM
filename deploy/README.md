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
4. Скрипт синкает runtime bundle и adapter на GPU-машину.
5. Скрипт активирует `REMOTE_VENV_ACTIVATE` и прогоняет короткий remote smoke.
6. Скрипт пишет `deploy/.runtime.prod.env`.
7. Скрипт поднимает контейнер через `deploy/docker-compose.prod.yml`.
8. Скрипт делает healthcheck URL.

## Где настраивается прод

Основные переменные лежат в:

```text
deploy/model-release.env
```

Ключевые поля:

- `APP_BACKEND=remote-hf`
- `APP_REPORT_DIR=artifacts/qwen25_1_5b_instruct_qlora_v1`
- `DEPLOY_PUBLIC_HOST=tinyllm.runningdog.org`
- `REMOTE_SSH_HOST=localhost`
- `REMOTE_SSH_PORT=2222`
- `REMOTE_SSH_USER=angel`
- `REMOTE_WORKDIR=/home/angel/projects/TinyLLM`
- `REMOTE_VENV_ACTIVATE=/home/angel/aider-env/bin/activate`
- `REMOTE_CONFIG=configs/qwen25_1_5b_instruct_qlora.toml`
- `REMOTE_ADAPTER=artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`
- `RUNTIME_REMOTE_SSH_HOST=localhost`
- `RUNTIME_SSH_DIR=/home/angel/.ssh`

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

3. Живо ли удаленное окружение:

```bash
ssh -p 2222 angel@localhost \
  'test -f /home/angel/aider-env/bin/activate'
```

4. Работает ли Docker Compose:

```bash
docker compose version
```

5. Открывается ли healthcheck URL после выкладки:

```bash
curl -I https://tinyllm.runningdog.org/
```

## Что обновлять при новой модели

1. Обновить remote target в `deploy/model-release.env`, если сменились пути.
2. Положить в git свежие report assets.
3. Положить в git финальный adapter в `artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`.
4. Сделать push в `main`.
