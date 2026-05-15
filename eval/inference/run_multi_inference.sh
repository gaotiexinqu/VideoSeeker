#!/bin/bash
#
# Multi-Benchmark Parallel Inference Script
#
# Features:
#   - Reads benchmarks.json config and runs N benchmark inference tasks in parallel
#   - Each benchmark uses an independent vLLM server on a different GPU and port
#   - All benchmarks share unified START_IDX / END_IDX
#   - Output path: {BASE_OUTPUT_DIR}/{MODEL_NAME}_{MODE}_{FPS}_{MAX_FRAMES}_{MAX_PIXELS}/{DATASET}.jsonl
#
# Adding a New Benchmark:
#   Simply add an entry in benchmarks.json, no need to modify this script
#
# Usage:
#   Modify the model config below and run directly:
#     bash run_multi_inference.sh
#

set -e

# Bypass proxy for localhost (required for vLLM local deployment)
export no_proxy=localhost,127.0.0.1

# ── Script directory ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1.  Change mode, tools etc. in bench
# mode:     choices=["direct", "reasoning", "tool"],

CKPT_PATH="[ckpt_path]"
IS_QWEN3_VL=True
GPU_UTIL=0.9
MAX_GPUS=8              # max number of GPUs to use (i.e. max parallel benchmarks)

# +++ Frame sampling parameters ──────────────────────────────────────────────────────
FPS=1
MAX_FRAMES=256
MAX_PIXELS=112896

# ── Unified inference range ─────────────────────────────────────────────────────────
START_IDX=0
END_IDX=-1               # unified control for all benchmarks; -1 means all data

# ── Output directory ────────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR="[base_output_dir]"
MODEL_NAME=$(basename "$CKPT_PATH")

# ── Check jq ───────────────────────────────────────────────────────────────────────
if ! command -v jq &> /dev/null; then
    echo "[ERROR] jq not installed. Install with: sudo apt install jq"
    exit 1
fi

# +++
# ── Load benchmark config ──────────────────────────────────────────────────────────
BENCHMARK_CONFIG="${SCRIPT_DIR}/benchmarks.json"
if [ ! -f "$BENCHMARK_CONFIG" ]; then
    echo "[ERROR] Benchmark config not found: $BENCHMARK_CONFIG"
    exit 1
fi

NUM_BENCH=$(jq length "$BENCHMARK_CONFIG")
if [ "$NUM_BENCH" -eq 0 ]; then
    echo "[ERROR] No benchmarks configured in benchmarks.json"
    exit 1
fi

if [ "$NUM_BENCH" -gt "$MAX_GPUS" ]; then
    echo "[ERROR] Number of benchmarks ($NUM_BENCH) exceeds MAX_GPUS ($MAX_GPUS). Adjust MAX_GPUS."
    exit 1
fi

echo "[INFO] Detected $NUM_BENCH benchmarks, using up to $MAX_GPUS GPUs"
echo "[INFO] MODEL_NAME: $MODEL_NAME"
echo "[INFO] Inference range: [$START_IDX, $END_IDX)"
echo ""

# ── Helper functions ────────────────────────────────────────────────────────────────

resolve_root() {
    local root="$1"
    echo "$root" | sed "s|\$ROOT|${ROOT}|g"
}

wait_for_server() {
    local port=$1
    local name=$2
    echo "[INFO] Waiting for $name vLLM server (port $port) to be ready (up to 2000s)..."
    MAX_WAIT=2000
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"data"'; then
            echo "[INFO] $name vLLM server ready (took ${WAITED}s)"
            return 0
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[INFO] Still waiting for $name... ${WAITED}s"
        fi
    done
    echo "[ERROR] $name vLLM server startup timeout (${MAX_WAIT}s)"
    return 1
}

# ── Step 1: Start vLLM servers (GPU 0 to N-1)───────────────────────────────────────
declare -a VLLM_PIDS=()
echo "========================================"
echo "  Step 1: Start vLLM servers"
echo "========================================"

for i in $(seq 0 $((NUM_BENCH - 1))); do
    GPU_ID=$i
    PORT=$((8000 + i))
    BENCH_NAME=$(jq -r ".[$i].name" "$BENCHMARK_CONFIG")

    echo ""
    echo "[INFO] Starting vLLM server [$i/$NUM_BENCH]: $BENCH_NAME"
    echo "[INFO]   GPU: $GPU_ID  Port: $PORT"

    # +++
    VLLM_LOG="${SCRIPT_DIR}/logs/vllm_serve_gpu${GPU_ID}.log"
    mkdir -p "${SCRIPT_DIR}/logs"
    VLLM_CMD="vllm serve $CKPT_PATH"
    VLLM_CMD+=" --tool-call-parser hermes"
    VLLM_CMD+=" --enable-auto-tool-choice"
    VLLM_CMD+=" --trust-remote-code"
    VLLM_CMD+=" --port $PORT"
    VLLM_CMD+=" --gpu-memory-utilization $GPU_UTIL"
    VLLM_CMD+=" --mm-processor-cache-gb 0"

    if [ "$IS_QWEN3_VL" != "True" ]; then
        VLLM_CMD+=" --chat-template ${SCRIPT_DIR}/tool_call_qwen2_5_vl.jinja"
    fi

    echo "[INFO] Command: $VLLM_CMD"
    echo "[INFO] Log: $VLLM_LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID $VLLM_CMD > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    VLLM_PIDS+=($VLLM_PID)
    echo "[INFO] vLLM server PID: $VLLM_PID"
done

# ── Step 2: Wait for all vLLM servers to be ready ───────────────────────────────────
echo ""
echo "========================================"
echo "  Step 2: Wait for all vLLM servers to be ready"
echo "========================================"

for i in $(seq 0 $((NUM_BENCH - 1))); do
    GPU_ID=$i
    PORT=$((8000 + i))
    BENCH_NAME=$(jq -r ".[$i].name" "$BENCHMARK_CONFIG")
    wait_for_server $PORT "$BENCH_NAME" || {
        echo "[ERROR] vLLM server failed to start, terminating all processes..."
        for pid in "${VLLM_PIDS[@]}"; do
            kill $pid 2>/dev/null || true
        done
        exit 1
    }
done

echo ""
echo "[INFO] All vLLM servers are ready!"
echo ""

# ── Step 3: Run inference tasks in parallel ────────────────────────────────────────
echo "========================================"
echo "  Step 3: Run inference tasks in parallel"
echo "========================================"

declare -a INFER_PIDS=()

for i in $(seq 0 $((NUM_BENCH - 1))); do
    GPU_ID=$i
    PORT=$((8000 + i))

    # Read config
    ROOT=$(jq -r ".[$i].root" "$BENCHMARK_CONFIG")
    FRAMES_ROOT=$(resolve_root "$(jq -r ".[$i].frames_root" "$BENCHMARK_CONFIG")")
    VIDEOS_ROOT=$(resolve_root "$(jq -r ".[$i].videos_root" "$BENCHMARK_CONFIG")")
    DATASET_INFO_PATH=$(resolve_root "$(jq -r ".[$i].dataset_info_path" "$BENCHMARK_CONFIG")")
    MEDIA_ROOT=$(resolve_root "$(jq -r ".[$i].media_root" "$BENCHMARK_CONFIG")")
    DATASET=$(jq -r ".[$i].name" "$BENCHMARK_CONFIG")
    TOOLS=$(jq -r ".[$i].tools" "$BENCHMARK_CONFIG")
    MODE=$(jq -r ".[$i].mode" "$BENCHMARK_CONFIG")
    DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${MODE}_${FPS}_${MAX_FRAMES}_${MAX_PIXELS}"
    mkdir -p "$DIR"
    echo "OUTPUT_DIR: ${DIR}"

    # Handle empty frames_root (some datasets don't need it)
    if [ "$FRAMES_ROOT" = "\$ROOT/frames" ] || [ "$FRAMES_ROOT" = "$" ]; then
        FRAMES_ROOT=""
    fi

    SAVE_PATH="${DIR}/${DATASET}.jsonl"

    echo ""
    echo "[INFO] Starting inference [$i/$NUM_BENCH]: $DATASET"
    echo "[INFO]   GPU: $GPU_ID  Port: $PORT"
    echo "[INFO]   DATASET_INFO_PATH: $DATASET_INFO_PATH"
    echo "[INFO]   VIDEOS_ROOT: $VIDEOS_ROOT"
    echo "[INFO]   FRAMES_ROOT: $FRAMES_ROOT"
    echo "[INFO]   TOOLS: ${TOOLS:-'(none)'}"
    echo "[INFO]   MODE: $MODE"
    echo "[INFO]   SAVE_PATH: $SAVE_PATH"

    INFER_LOG="${SCRIPT_DIR}/logs/inference_gpu${GPU_ID}.log"

    # Build inference command
    PYTHON_CMD="python ${SCRIPT_DIR}/single_inference.py"
    PYTHON_CMD+=" --dataset \"$DATASET\""
    PYTHON_CMD+=" --dataset_info_path \"$DATASET_INFO_PATH\""
    PYTHON_CMD+=" --videos_root \"$VIDEOS_ROOT\""
    PYTHON_CMD+=" --media_root \"$MEDIA_ROOT\""
    if [ -n "$FRAMES_ROOT" ]; then
        PYTHON_CMD+=" --frames_root \"$FRAMES_ROOT\""
    fi
    PYTHON_CMD+=" --save_path \"$SAVE_PATH\""
    PYTHON_CMD+=" --start_idx $START_IDX"
    PYTHON_CMD+=" --end_idx $END_IDX"
    PYTHON_CMD+=" --api_base \"http://localhost:$PORT/v1\""
    PYTHON_CMD+=" --fps $FPS"
    PYTHON_CMD+=" --max_frames $MAX_FRAMES"
    PYTHON_CMD+=" --max_pixels $MAX_PIXELS"
    PYTHON_CMD+=" --mode \"$MODE\""
    if [ "$MODE" = "tool" ] && [ -n "$TOOLS" ]; then
        PYTHON_CMD+=" --tools \"$TOOLS\""
    fi

    echo "[INFO] Command: $PYTHON_CMD"
    echo "[INFO] Log: $INFER_LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "$PYTHON_CMD" > "$INFER_LOG" 2>&1 &
    INFER_PID=$!
    INFER_PIDS+=($INFER_PID)
    echo "[INFO] Inference PID: $INFER_PID"
done

# ── Step 4: Wait for all inference tasks to complete ───────────────────────────────
echo ""
echo "========================================"
echo "  Step 4: Wait for all inference tasks to complete"
echo "========================================"
echo "[INFO] ${#INFER_PIDS[@]} inference tasks running in background"
echo ""

FAILED=0
for i in "${!INFER_PIDS[@]}"; do
    PID=${INFER_PIDS[$i]}
    BENCH_NAME=$(jq -r ".[$i].name" "$BENCHMARK_CONFIG")
    GPU_ID=$i

    if wait $PID; then
        echo "[INFO] [$BENCH_NAME] inference completed (PID: $PID)"
    else
        echo "[ERROR] [$BENCH_NAME] inference failed (PID: $PID)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "  Inference complete"
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo "[INFO] All ${#INFER_PIDS[@]} tasks completed successfully!"
else
    echo "[WARN] $FAILED task(s) failed. Check logs."
fi

echo ""
echo "[INFO] Result files:"
for i in $(seq 0 $((NUM_BENCH - 1))); do
    DATASET=$(jq -r ".[$i].name" "$BENCHMARK_CONFIG")
    MODE=$(jq -r ".[$i].mode" "$BENCHMARK_CONFIG")
    SAVE_PATH="${DIR}/${DATASET}.jsonl"
    echo "  - $SAVE_PATH"
done

echo "[INFO] Successfully completed inference for all benchmarks."
# ── Cleanup vLLM servers (uncomment when needed) ───────────────────────────────────
# echo "[INFO] Stopping vLLM servers..."
# for pid in "${VLLM_PIDS[@]}"; do
#     kill $pid 2>/dev/null || true
# done
# echo "[INFO] Done."
