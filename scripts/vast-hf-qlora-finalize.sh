#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  scripts/vast-hf-qlora-finalize.sh INSTANCE_ID [RUN_NAME] [CONFIG_PATH]

Purpose:
  1. Run a quick remote smoke-test against the finished adapter
  2. Download artifacts and train log to a timestamped local backup dir
  3. Build a local training plot from the downloaded run
  4. Destroy the Vast instance

Defaults:
  RUN_NAME    qwen25_1_5b_instruct_qlora_v1
  CONFIG_PATH configs/qwen25_1_5b_instruct_qlora.toml

Environment:
  VAST_SSH_KEY       SSH private key, default: ~/.ssh/vast_ed25519
  VAST_REMOTE_ROOT   Project root on remote machine, default: /workspace/TinyLLM
  VAST_BACKUP_ROOT   Local backup root, default: backups/vast
  VAST_PROMPTS_FILE  Prompt set for smoke test, default: data/evals/base-vs-lora-prompts.jsonl
  VAST_SMOKE_LIMIT   Number of prompts for the remote smoke test, default: 4
  VAST_DESTROY       true or false, default: true

Notes:
  - If the remote smoke-test or artifact download fails, the script exits before destroy.
  - Local plotting runs after download, when artifacts are already safe on the local machine.
EOF
  exit 1
fi

instance_id="$1"
run_name="${2:-qwen25_1_5b_instruct_qlora_v1}"
config_path="${3:-configs/qwen25_1_5b_instruct_qlora.toml}"

ssh_key="${VAST_SSH_KEY:-$HOME/.ssh/vast_ed25519}"
remote_root="${VAST_REMOTE_ROOT:-/workspace/TinyLLM}"
backup_root="${VAST_BACKUP_ROOT:-backups/vast}"
prompts_file="${VAST_PROMPTS_FILE:-data/evals/base-vs-lora-prompts.jsonl}"
smoke_limit="${VAST_SMOKE_LIMIT:-4}"
destroy_instance="${VAST_DESTROY:-true}"

ssh_url="$(vastai ssh-url "$instance_id" | tr -d '\r\n')"
host_port="${ssh_url#ssh://root@}"
ssh_host="${host_port%:*}"
ssh_port="${host_port##*:}"

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_dir="${backup_root}/${timestamp}-${run_name}"
remote_run_dir="${remote_root}/artifacts/${run_name}"
remote_log_path="${remote_root}/logs/${run_name}_train.log"

ssh_cmd=(
  ssh
  -i "$ssh_key"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "$ssh_port"
  "root@${ssh_host}"
)

scp_cmd=(
  scp
  -i "$ssh_key"
  -P "$ssh_port"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

echo "Remote smoke test on instance ${instance_id} (${ssh_host}:${ssh_port})"
"${ssh_cmd[@]}" \
  "cd '${remote_root}' \
  && export PATH=\"\$HOME/.local/bin:\$PATH\" \
  && export PYTHONPATH='${remote_root}/src' \
  && uv run tinyllm-hf-smoke \
      --config '${config_path}' \
      --adapter 'artifacts/${run_name}/adapter' \
      --prompts-file '${prompts_file}' \
      --limit '${smoke_limit}' \
      --output 'artifacts/${run_name}/smoke/posttrain_smoke.jsonl'"

echo "Downloading artifacts to ${backup_dir}"
mkdir -p "$backup_dir"
"${scp_cmd[@]}" -r "root@${ssh_host}:${remote_run_dir}" "$backup_dir/"
"${scp_cmd[@]}" "root@${ssh_host}:${remote_log_path}" "$backup_dir/"

if [[ "$destroy_instance" == "true" ]]; then
  echo "Destroying Vast instance ${instance_id}"
  vastai destroy instance "$instance_id"
else
  echo "Skipping destroy because VAST_DESTROY=${destroy_instance}"
fi

echo "Building local plot from downloaded run"
uv run tinyllm-hf-plot --run-dir "${backup_dir}/${run_name}"

echo "Done."
echo "Backup dir: ${backup_dir}"
echo "Smoke file: ${backup_dir}/${run_name}/smoke/posttrain_smoke.jsonl"
echo "Plot: ${backup_dir}/${run_name}/plots/training_history.png"
