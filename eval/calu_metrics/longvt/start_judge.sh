#!/bin/bash
# Deploy vLLM Judge model and evaluate a single jsonl file
# Usage: bash start_judge.sh <JSONL_FILE> [OUTPUT_FILE]
#   JSONL_FILE: Path to the jsonl file
#   OUTPUT_FILE: Output file path (default: {JSONL_BASENAME}_llm.jsonl)

set -e

# Bypass proxy for localhost
export no_proxy=localhost,127.0.0.1

# Config
GPU_ID=0
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=1235
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.60
MODEL_PATH="judge"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluation.py"

INPUT_JSONL="/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/examples_longvt/eval/outputs/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b_tool_1_128_200704/VideoSIAH-Eval.jsonl"


LOG_FILE="/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/examples_longvt/eval/calu_metrics/longvt/vllm_judge.log"
if [ ! -f "$INPUT_JSONL" ]; then
    echo "[ERROR] File not found: $INPUT_JSONL"
    exit 1
fi

if [ -n "$2" ]; then
    OUTPUT_FILE="$2"
else
    OUTPUT_FILE="${INPUT_JSONL%.jsonl}_llm.jsonl"
fi

BASENAME=$(basename "$INPUT_JSONL" .jsonl)

echo "========================================="
echo "  Evaluated file: $(basename "$INPUT_JSONL")"
echo "  Benchmark: $BASENAME"
echo "  Output: $(basename "$OUTPUT_FILE")"
echo "========================================="
echo ""

# Allow overriding max model length (needed for some scenarios)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ── Start vLLM Service ─────────────────────────────────────────────────────────
echo "[1/3] Starting vLLM service..."

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Start vLLM service (background)
nohup vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --served-model-name $MODEL_PATH \
    --trust-remote-code \
    --enforce-eager \
    > ${LOG_FILE} 2>&1 &

VLLM_PID=$!
echo "  vLLM process PID: $VLLM_PID"

# Health check
echo "[2/3] Waiting for service to be ready..."
MAX_RETRIES=120
RETRY_INTERVAL=10
HEALTH_URL="http://localhost:$PORT/health"

for i in $(seq 1 $MAX_RETRIES); do
    if curl -s --noproxy "*" --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "  Service ready (attempt $i/$MAX_RETRIES)"
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        echo "  Service startup timed out, check vllm_judge.log"
        exit 1
    fi

    echo "  Waiting... ($i/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done

# Validate API
echo "[3/3] Validating API availability..."
curl -s --noproxy "*" --max-time 30 -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_PATH'",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }' > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  API validation successful!"
    echo "  Judge service address: http://localhost:$PORT/v1"
    echo "  Model name: $MODEL_PATH"
    echo "  Log: vllm_judge.log"
    echo ""
else
    echo "  API validation failed, check logs"
    exit 1
fi

# ── Run Evaluation ─────────────────────────────────────────────────────────────
echo "Starting evaluation: $INPUT_JSONL -> $(basename "$OUTPUT_FILE")"

# Bypass proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

# Set Judge API address (matches the vLLM service port started above)
export JUDGE_API_BASE="http://localhost:$PORT/v1"

python $EVAL_SCRIPT --input "$INPUT_JSONL" --output "$OUTPUT_FILE"

echo "Done: $(basename "$OUTPUT_FILE")"

echo "========================================="
echo "Evaluation complete!"
echo "========================================="
