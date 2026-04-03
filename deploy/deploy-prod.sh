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

echo "Loading deploy configuration from deploy/model-release.env"

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
  : "${REMOTE_VENV_ACTIVATE:?REMOTE_VENV_ACTIVATE must be set for remote-hf backend}"
  : "${REMOTE_CONFIG:?REMOTE_CONFIG must be set for remote-hf backend}"
  : "${REMOTE_ADAPTER:?REMOTE_ADAPTER must be set for remote-hf backend}"

  required_local_files+=(
    "${APP_REPORT_DIR}/hf_train_setup.json"
    "${APP_REPORT_DIR}/plots/training_history.png"
    "${APP_REPORT_DIR}/plots/plot_summary.json"
    "${APP_REPORT_DIR}/smoke/posttrain_smoke.jsonl"
    "${APP_REPORT_DIR}/adapter/adapter_config.json"
    "${APP_REPORT_DIR}/adapter/adapter_model.safetensors"
    "${APP_REPORT_DIR}/adapter/tokenizer.json"
    "${APP_REPORT_DIR}/adapter/tokenizer_config.json"
  )
fi

for path in "${required_local_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required local file is missing: $path" >&2
    exit 1
  fi
done

healthcheck_url="${DEPLOY_HEALTHCHECK_URL:-https://tinyllm.runningdog.org/}"
deploy_public_host="${DEPLOY_PUBLIC_HOST:-}"
if [[ -z "${deploy_public_host}" ]]; then
  deploy_public_host="$(printf '%s' "${healthcheck_url}" | sed -E 's#https?://([^/]+)/?.*#\1#')"
fi
runtime_remote_ssh_host="${RUNTIME_REMOTE_SSH_HOST:-${REMOTE_SSH_HOST:-}}"
runtime_ssh_dir="${RUNTIME_SSH_DIR:-${HOME}/.ssh}"
runtime_env_path="deploy/.runtime.prod.env"

if [[ "${APP_BACKEND}" == "remote-hf" ]]; then
  remote_target="${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}"
  ssh_opts=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -p "${REMOTE_SSH_PORT}"
  )
  remote_config_path="${REMOTE_CONFIG}"
  remote_adapter_path="${REMOTE_ADAPTER}"
  if [[ "${REMOTE_CONFIG}" != /* ]]; then
    remote_config_path="${REMOTE_WORKDIR}/${REMOTE_CONFIG}"
  fi
  if [[ "${REMOTE_ADAPTER}" != /* ]]; then
    remote_adapter_path="${REMOTE_WORKDIR}/${REMOTE_ADAPTER}"
  fi

  echo "Preparing remote runtime on ${remote_target}:${REMOTE_WORKDIR}"
  ssh "${ssh_opts[@]}" "${remote_target}" "mkdir -p '${REMOTE_WORKDIR}'"

  echo "Syncing TinyLLM runtime bundle to GPU host"
  rsync -az --delete --relative \
    -e "ssh ${ssh_opts[*]}" \
    configs \
    src \
    pyproject.toml \
    uv.lock \
    "${APP_REPORT_DIR}" \
    "${remote_target}:${REMOTE_WORKDIR}/"

  echo "Validating remote Python environment and model load"
  ssh "${ssh_opts[@]}" "${remote_target}" bash -s -- \
    "${REMOTE_WORKDIR}" \
    "${REMOTE_VENV_ACTIVATE}" \
    "${remote_config_path}" \
    "${remote_adapter_path}" <<'EOF'
set -euo pipefail

remote_workdir="$1"
remote_activate="$2"
remote_config="$3"
remote_adapter="$4"

test -d "$remote_workdir"
test -f "$remote_activate"
test -f "$remote_config"
test -d "$remote_adapter"

source "$remote_activate"
export PYTHONPATH="$remote_workdir/src"

python - <<'PY'
import importlib.util

required = ["torch", "transformers", "peft", "bitsandbytes", "sentencepiece"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing remote Python packages: {', '.join(missing)}")
PY

python -m tinyllm.hf_generate \
  --config "$remote_config" \
  --adapter "$remote_adapter" \
  --prompt "Коротко представься по-русски." \
  --max-new-tokens 24 \
  --temperature 0.2 \
  --top-p 0.9 \
  >/tmp/tinyllm-remote-smoke.txt
EOF
fi

if [[ ! -d "${runtime_ssh_dir}" ]]; then
  echo "Runtime SSH directory is missing: ${runtime_ssh_dir}" >&2
  exit 1
fi

cat > "${runtime_env_path}" <<EOF
APP_BACKEND=${APP_BACKEND}
APP_CONFIG=${APP_CONFIG}
APP_HOST=${APP_HOST}
APP_PORT=${APP_PORT}
APP_REPORT_DIR=${APP_REPORT_DIR}
REMOTE_SSH_HOST=${runtime_remote_ssh_host}
REMOTE_SSH_PORT=${REMOTE_SSH_PORT}
REMOTE_SSH_USER=${REMOTE_SSH_USER}
REMOTE_WORKDIR=${REMOTE_WORKDIR}
REMOTE_VENV_ACTIVATE=${REMOTE_VENV_ACTIVATE}
REMOTE_CONFIG=${REMOTE_CONFIG}
REMOTE_ADAPTER=${REMOTE_ADAPTER}
REMOTE_MAX_NEW_TOKENS=${REMOTE_MAX_NEW_TOKENS}
REMOTE_TEMPERATURE=${REMOTE_TEMPERATURE}
REMOTE_TOP_K=${REMOTE_TOP_K}
DEPLOY_PUBLIC_HOST=${deploy_public_host}
RUNTIME_SSH_DIR=${runtime_ssh_dir}
EOF

chmod 600 "${runtime_env_path}"

echo "Building and starting production container from current checkout"
docker rm -f tinyllm >/dev/null 2>&1 || true
docker compose --env-file "${runtime_env_path}" -f deploy/docker-compose.prod.yml up -d --build --force-recreate

echo "Waiting for healthcheck ${healthcheck_url}"
curl \
  --fail \
  --silent \
  --show-error \
  --location \
  --retry 10 \
  --retry-all-errors \
  --retry-delay 3 \
  --max-time 30 \
  "${healthcheck_url}" >/dev/null

echo "Deploy completed successfully."
