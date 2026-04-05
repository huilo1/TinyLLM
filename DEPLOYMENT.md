# Deployment

Этот репозиторий деплоится в `https://tinyllm.runningdog.org/` через push в `main`.

## Источник истины

Авторитетный deploy entrypoint в репозитории:

```text
deploy/deploy-prod.sh
```

GitHub Actions сначала checkout'ит текущий commit, затем запускает именно этот скрипт. Весь deploy выполняется из текущего checkout. Внешний агент больше не должен “догадываться”, что делать. Последовательность шагов описана и зафиксирована в репозитории.

## Текущая прод-схема

- вебморда и reverse-proxy живут на deploy-сервере;
- inference не выполняется на веб-сервере;
- docker-контейнер с UI поднимается из текущего commit через `deploy/docker-compose.prod.yml` в `network_mode: host`;
- вебморда держит SSH-backed worker и отправляет запросы на GPU-машину:
  - host с точки зрения deploy-сервера: `localhost`
  - host с точки зрения контейнера: `localhost`
  - port: `2222`
  - user: `angel`
- remote worker стартует лениво: если после перезагрузки GPU-хоста процесса нет, первый запрос из вебки сам поднимает его через SSH и ждет cold start;
- обучение-репорты лежат прямо в git как lightweight assets:
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/training_history.png`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/plot_summary.json`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/hf_train_setup.json`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/smoke/posttrain_smoke.jsonl`

## Что делает deploy script

`deploy/deploy-prod.sh`:

1. Загружает `deploy/model-release.env`.
2. Проверяет наличие обязательных локальных файлов для UI и training report.
3. Для `APP_BACKEND=remote-hf` подключается по SSH к GPU-хосту и создает `REMOTE_WORKDIR`.
4. Синхронизирует на GPU-хост минимальный runtime bundle из текущего commit:
   - `configs/`
   - `src/`
   - `pyproject.toml`
   - `uv.lock`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/`
5. Активирует удаленное окружение из `REMOTE_VENV_ACTIVATE` и делает remote smoke:
   - проверяет импорты `torch/transformers/peft/bitsandbytes/sentencepiece`
   - запускает короткую генерацию через `tinyllm.hf_generate`
6. Пишет runtime env в `deploy/.runtime.prod.env`.
7. Собирает и поднимает контейнер через `docker compose -f deploy/docker-compose.prod.yml`.
8. Делает healthcheck URL после деплоя.

Если один из preflight-checks не проходит, deploy должен падать до выкладки.

## Обязательные предпосылки на прод-сервере

- self-hosted GitHub runner с label `tinyllm`
- доступен Docker и `docker compose`
- работает безпарольный SSH от deploy-сервера к `angel@localhost -p 2222`
- на GPU-хосте уже существует и готово к использованию Python-окружение:
  - `REMOTE_VENV_ACTIVATE=/home/angel/aider-env/bin/activate`
- на deploy-сервере есть SSH-ключи для runtime контейнера:
  - `RUNTIME_SSH_DIR=/home/angel/.ssh`
- в текущем commit уже лежат:
  - report assets в `artifacts/qwen25_1_5b_instruct_qlora_v1`
  - финальный QLoRA adapter в `artifacts/qwen25_1_5b_instruct_qlora_v1/adapter`

## Как обновлять модель

При новой модели нужно обновить две вещи:

1. Remote inference target:
   - `deploy/model-release.env`
   - при необходимости `REMOTE_ADAPTER`, `REMOTE_CONFIG`, `REMOTE_VENV_ACTIVATE`, `RUNTIME_REMOTE_SSH_HOST`, generation defaults

2. Репорты обучения:
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/*`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/hf_train_setup.json`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/smoke/posttrain_smoke.jsonl`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/adapter/*`

После этого достаточно push в `main`.
