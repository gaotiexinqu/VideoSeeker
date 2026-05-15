#!/bin/bash

ulimit -n 65536 2>/dev/null
ulimit -u 65536 2>/dev/null

DATE=$(date +%m%d_%H%M)

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

source activate /mnt/tidal-alsh01/dataset/zeus/zhaoy/envs/llama-factory

unset WANDB_DISABLED
wandb login "[token]"
export WANDB_PROJECT="[project_name]"
export WANDB_RUN_NAME="[run_name]"
export WANDB_MODE=online

WORK_DIR="/path/to/work_dir"
YAML_CONFIG="${WORK_DIR}/qwen3vl_full_sft_non_streaming.yaml"
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_${DATE}.log"

mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

llamafactory-cli train "${YAML_CONFIG}" 2>&1 | tee "${LOG_FILE}"