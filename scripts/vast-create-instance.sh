#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  scripts/vast-create-instance.sh OFFER_ID [extra vastai args...]

Environment:
  VAST_IMAGE        Docker image to launch
                    default: ghcr.io/huilo1/tinyllm:cu124-v2
  VAST_DISK_GB      Instance disk size in GB
                    default: 40
  VAST_RUN_TYPE     ssh or jupyter
                    default: ssh
  VAST_DIRECT       true or false
                    default: true
  VAST_ENV          Raw value passed to --env
  VAST_ONSTART_CMD  Raw value passed to --onstart-cmd

Example:
  VAST_IMAGE=ghcr.io/huilo1/tinyllm:cu124-v2 \
  scripts/vast-create-instance.sh 1234567
EOF
  exit 1
fi

offer_id="$1"
shift

image="${VAST_IMAGE:-ghcr.io/huilo1/tinyllm:cu124-v2}"
disk_gb="${VAST_DISK_GB:-40}"
run_type="${VAST_RUN_TYPE:-ssh}"
direct="${VAST_DIRECT:-true}"

cmd=(
  vastai create instance
  "$offer_id"
  --image "$image"
  --disk "$disk_gb"
)

case "$run_type" in
  ssh)
    cmd+=(--ssh)
    ;;
  jupyter)
    cmd+=(--jupyter)
    ;;
  *)
    echo "Unsupported VAST_RUN_TYPE: $run_type" >&2
    exit 2
    ;;
esac

if [[ "$direct" == "true" ]]; then
  cmd+=(--direct)
fi

if [[ -n "${VAST_ENV:-}" ]]; then
  cmd+=(--env "$VAST_ENV")
fi

if [[ -n "${VAST_ONSTART_CMD:-}" ]]; then
  cmd+=(--onstart-cmd "$VAST_ONSTART_CMD")
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
