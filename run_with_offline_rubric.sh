#!/bin/bash
set -euo pipefail

# =============================================================================
# Pre-generate general rubrics offline, then launch policy training.
#
# Stage 1 fills `requirements_general` / `weights_general` columns on the
# parquet at `cfg.data.train_files` (in place; rows that already have a seed
# are skipped unless `data_generator.offline_overwrite_existing=true`).
#
# Stage 2 runs the trainer against the same config. AdaptiveDataset's init
# check refuses to start if any active row lacks a seed, so an error in
# Stage 1 fails the whole script before we burn GPU time on training.
#
# Both stages share the same Hydra config path/name and forward any extra
# arguments (`hydra` overrides) so you can do e.g.
#   ./run_with_offline_rubric.sh \
#       --config-name llm_tutor_base \
#       data.train_files=./datasets/wildchecklists/train.parquet
#
# CUDA_VISIBLE_DEVICES is inherited from the calling shell.
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

CONFIG_PATH="${PROJECT_DIR}/configs"
CONFIG_NAME=""
EXTRA=()

usage() {
    cat <<EOF
Usage: $0 --config-name <name> [--config-path <dir>] [hydra overrides...]

  --config-name   Hydra config name (without .yaml). Required.
  --config-path   Directory containing the config. Defaults to ${PROJECT_DIR}/configs.

  Any further args are forwarded verbatim to both stages as Hydra overrides.

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 --config-name llm_tutor
  CUDA_VISIBLE_DEVICES=2,3,6,7 $0 --config-name llm_tutor_base \\
      data_generator.offline_overwrite_existing=true
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-path)
            CONFIG_PATH="$2"; shift 2 ;;
        --config-name)
            CONFIG_NAME="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            EXTRA+=("$1"); shift ;;
    esac
done

if [[ -z "${CONFIG_NAME}" ]]; then
    echo "[error] --config-name is required" >&2
    usage >&2
    exit 2
fi

if [[ ! -d "${CONFIG_PATH}" ]]; then
    echo "[error] --config-path is not a directory: ${CONFIG_PATH}" >&2
    exit 2
fi

echo "================================================================================"
echo "Stage 1/2 — Offline general rubric pre-generation"
echo "  config-path: ${CONFIG_PATH}"
echo "  config-name: ${CONFIG_NAME}"
echo "  extra args:  ${EXTRA[*]:-<none>}"
echo "================================================================================"
uv run python -m data_generation.offline_general_rubric \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    "${EXTRA[@]}"

echo
echo "================================================================================"
echo "Stage 2/2 — Policy training"
echo "================================================================================"
uv run python -m verl.trainer.main_ppo \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    "${EXTRA[@]}"
