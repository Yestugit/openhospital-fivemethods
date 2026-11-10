#!/usr/bin/env bash
# Run the real public CLI. No provider fixtures or algorithm replacements.
set -euo pipefail
umask 077

if (( $# < 3 || $# > 5 )); then
    printf 'Usage: bash test/cli_e2e/run.sh LABEL CONFIG_JSON DATA_JSON [experience|streaming] [SPLIT]\n' >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
label=$1
if [[ ! $label =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'LABEL may contain only letters, numbers, underscore and dash.\n' >&2
    exit 2
fi
config_path="$repo_dir/test/cli_e2e/$2"
data_path="$repo_dir/test/cli_e2e/$3"
protocol=${4:-experience}
split=${5:-test}
: "${SILICONFLOW_API_KEY:?Load the SiliconFlow credential into the environment first}"
output_root=${MEMATLAS_CLI_E2E_ROOT:-"$repo_dir/.mematlas/runs/cli-e2e-$(date +%Y%m%d-%H%M%S)"}
mkdir -p -- "$output_root"
output_root=$(cd -- "$output_root" && pwd)
report_dir="$output_root/$label"
if [[ -e $report_dir ]]; then
    printf 'Refusing to overwrite an existing run: %s\n' "$report_dir" >&2
    exit 2
fi
mkdir -p -- "$report_dir/workdir"
export MEMATLAS_E2E_REPO="$repo_dir"
export PYTHONPATH="$repo_dir/test/cli_e2e/instrumentation:$(dirname -- "$repo_dir")"
# MemCon keeps its native CPU embedding model; use the existing model cache.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
cd -- "$report_dir/workdir"
set +e
timeout 360s "$repo_dir/.venv/bin/python" -m mematlas run \
    --config "$config_path" --benchmark custom --data "$data_path" \
    --protocol "$protocol" --train-split train --split "$split" \
    --report-dir "$report_dir" \
    --model Qwen/Qwen2.5-72B-Instruct \
    --base-url https://api.siliconflow.cn/v1 \
    --api-key-env SILICONFLOW_API_KEY > "$report_dir/cli.log" 2>&1
cli_status=$?
set -e
printf 'CLI exit_code=%s report=%s\n' "$cli_status" "$report_dir"
exit "$cli_status"
