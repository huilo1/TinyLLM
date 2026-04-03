#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f deploy/model-release.env ]]; then
  echo "deploy/model-release.env not found" >&2
  exit 1
fi

set -a
source deploy/model-release.env
set +a

: "${APP_BACKEND:?APP_BACKEND must be set in deploy/model-release.env}"
: "${APP_CONFIG:?APP_CONFIG must be set in deploy/model-release.env}"

required_local_files=(
  "$APP_CONFIG"
  "src/tinyllm/app.py"
  "src/tinyllm/remote_inference.py"
  "src/tinyllm/hf_worker.py"
)

if [[ "${APP_BACKEND}" == "remote-hf" ]]; then
  : "${APP_REPORT_DIR:?APP_REPORT_DIR must be set for remote-hf backend}"
  : "${REMOTE_SSH_HOST:?REMOTE_SSH_HOST must be set for remote-hf backend}"
  : "${REMOTE_SSH_PORT:?REMOTE_SSH_PORT must be set for remote-hf backend}"
  : "${REMOTE_SSH_USER:?REMOTE_SSH_USER must be set for remote-hf backend}"
  : "${REMOTE_WORKDIR:?REMOTE_WORKDIR must be set for remote-hf backend}"
  : "${REMOTE_CONFIG:?REMOTE_CONFIG must be set for remote-hf backend}"
  : "${REMOTE_ADAPTER:?REMOTE_ADAPTER must be set for remote-hf backend}"

  required_local_files+=(
    "${APP_REPORT_DIR}/hf_train_setup.json"
    "${APP_REPORT_DIR}/plots/training_history.png"
    "${APP_REPORT_DIR}/plots/plot_summary.json"
    "${APP_REPORT_DIR}/smoke/posttrain_smoke.jsonl"
  )
fi

for path in "${required_local_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required local file is missing: $path" >&2
    exit 1
  fi
done

if [[ "${APP_BACKEND}" == "remote-hf" ]]; then
  remote_target="${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}"
  remote_config_path="${REMOTE_CONFIG}"
  remote_adapter_path="${REMOTE_ADAPTER}"
  if [[ "${REMOTE_CONFIG}" != /* ]]; then
    remote_config_path="${REMOTE_WORKDIR}/${REMOTE_CONFIG}"
  fi
  if [[ "${REMOTE_ADAPTER}" != /* ]]; then
    remote_adapter_path="${REMOTE_WORKDIR}/${REMOTE_ADAPTER}"
  fi

  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -p "${REMOTE_SSH_PORT}" \
    "${remote_target}" \
    "test -d '${REMOTE_WORKDIR}' && test -f '${remote_config_path}' && test -d '${remote_adapter_path}'"
fi

if [[ ! -x /opt/apps/deploy-tinyllm.sh ]]; then
  echo "Platform deploy script /opt/apps/deploy-tinyllm.sh is missing or not executable" >&2
  exit 1
fi

/opt/apps/deploy-tinyllm.sh prod

healthcheck_url="${DEPLOY_HEALTHCHECK_URL:-https://tinyllm.runningdog.org/}"
curl --fail --silent --show-error --location --max-time 30 "${healthcheck_url}" >/dev/null

echo "Deploy completed successfully."
