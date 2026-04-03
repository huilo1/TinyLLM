# Deployment

Этот репозиторий деплоится в `https://tinyllm.runningdog.org/` через push в `main`.

## Источник истины

Авторитетный deploy entrypoint в репозитории:

```text
deploy/deploy-prod.sh
```

GitHub Actions сначала checkout'ит текущий commit, затем запускает именно этот скрипт. Уже внутри него вызывается platform-specific wrapper:

```text
/opt/apps/deploy-tinyllm.sh prod
```

То есть внешний агент больше не должен “догадываться”, что делать. Последовательность шагов описана и зафиксирована в репозитории.

## Текущая прод-схема

- вебморда и reverse-proxy живут на deploy-сервере;
- inference не выполняется на веб-сервере;
- вебморда держит SSH-backed worker и отправляет запросы на GPU-машину:
  - host: `localhost`
  - port: `2222`
  - user: `angel`
- обучение-репорты лежат прямо в git как lightweight assets:
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/training_history.png`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/plot_summary.json`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/hf_train_setup.json`
  - `artifacts/qwen25_1_5b_instruct_qlora_v1/smoke/posttrain_smoke.jsonl`

## Что делает deploy script

`deploy/deploy-prod.sh`:

1. Загружает `deploy/model-release.env`.
2. Проверяет наличие обязательных локальных файлов для UI и training report.
3. Для `APP_BACKEND=remote-hf` проверяет SSH-доступ и существование:
   - `REMOTE_WORKDIR`
   - `REMOTE_CONFIG`
   - `REMOTE_ADAPTER`
4. Запускает platform deploy wrapper `/opt/apps/deploy-tinyllm.sh prod`.
5. Делает healthcheck URL после деплоя.

Если один из preflight-checks не проходит, deploy должен падать до выкладки.

## Обязательные предпосылки на прод-сервере

- self-hosted GitHub runner с label `tinyllm`
- доступен `/opt/apps/deploy-tinyllm.sh`
- работает безпарольный SSH от deploy-сервера к `angel@localhost -p 2222`
- на GPU-хосте уже лежат:
  - repo в `REMOTE_WORKDIR`
  - QLoRA adapter в `REMOTE_ADAPTER`
  - config в `REMOTE_CONFIG`

## Как обновлять модель

При новой модели нужно обновить две вещи:

1. Remote inference target:
   - `deploy/model-release.env`
   - при необходимости `REMOTE_ADAPTER`, `REMOTE_CONFIG`, generation defaults

2. Репорты обучения:
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/plots/*`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/hf_train_setup.json`
   - `artifacts/qwen25_1_5b_instruct_qlora_v1/smoke/posttrain_smoke.jsonl`

После этого достаточно push в `main`.
