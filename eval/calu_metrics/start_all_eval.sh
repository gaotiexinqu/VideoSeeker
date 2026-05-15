#!/bin/bash
#
# One-click evaluation for all benchmarks
#
# Usage:
#   bash 0_start_all_eval.sh <MODEL_DIR>
#   bash 0_start_all_eval.sh Qwen3-VL-4B-Instruct_mix6set_22k_reasoning_1_64_200704
#   bash 0_start_all_eval.sh /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/examples_longvt/eval/outputs/Qwen3-VL-4B-Instruct_mix6set_22k_reasoning_1_64_200704
#
# Notes:
#   Pass in a model directory path; script scans all .jsonl files under it.
#   Dir name format: {MODEL_NAME}_{MODE}_{FPS}_{MAX_FRAMES}_{MAX_PIXELS}
#   File name format: {DATASET}.jsonl (no MODE in filename)
#   MODE is extracted from dir name; benchmark name is extracted from file name.
#   VideoSIAH-Eval is skipped (open-ended QA, requires separate evaluation logic).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT_DIR="/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/examples_longvt/eval/outputs"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_all.py"

# New format: dir name contains MODE, file name does not contain MODE
# Dir: {MODEL_NAME}_{MODE}_{FPS}_{MAX_FRAMES}_{MAX_PIXELS}. 1_64_200704 values are not used; extra notes at the end are allowed.
INPUT="/path/to/inference_folder"

if [ -d "$INPUT" ]; then
    MODEL_DIR="$INPUT"
else
    MODEL_DIR="${BASE_OUTPUT_DIR}/${INPUT}"
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] Model dir not found: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] Eval script not found: $EVAL_SCRIPT"
    exit 1
fi

echo "========================================"
echo "  Start evaluation: $(basename "$MODEL_DIR")"
echo "  Dir: $MODEL_DIR"
echo "========================================"
echo ""

# ── Scan all JSONL files (filename has no MODE) ────────────────────────────────
cd "$MODEL_DIR"
RESULT_FILES=$(ls *.jsonl 2>/dev/null || true)

if [ -z "$RESULT_FILES" ]; then
    echo "[ERROR] No .jsonl files found in $MODEL_DIR"
    exit 1
fi

# ── Supported benchmarks list (for validation) ────────────────────────────────
SUPPORTED_BENCHS="V2P-Bench VideoMME LongVideoBench VideoReferSuit"

COUNTER=0
declare -a BENCH_FULL_NAMES=()
declare -a BENCH_MODES=()
declare -a BENCH_ACCURACIES=()
declare -a BENCH_EXTRA=()

for jsonl_file in $RESULT_FILES; do
    FULL_PATH="${MODEL_DIR}/${jsonl_file}"

    # New format: dir name contains MODE, file name does not contain MODE
    # Dir: {MODEL_NAME}_{MODE}_{FPS}_{MAX_FRAMES}_{MAX_PIXELS}
    # File: {DATASET}.jsonl
    MODEL_DIR_BASENAME=$(basename "$MODEL_DIR")
    # Extract MODE from dir name (after first underscore, must be direct/reasoning/tool)
    MODE=$(echo "$MODEL_DIR_BASENAME" | sed 's/.*_\(direct\|reasoning\|tool\)_.*/\1/')
    # Fall back to full dir name _direct/_reasoning/_tool if extraction fails
    if [ "$MODE" = "$MODEL_DIR_BASENAME" ]; then
        MODE=$(echo "$MODEL_DIR_BASENAME" | sed -E 's/.*_(direct|reasoning|tool).*/\1/')
    fi
    # Benchmark name is the file name (strip .jsonl suffix)
    BENCH_NAME="${jsonl_file%.jsonl}"

    echo "----------------------------------------"
    echo "  Found file: $jsonl_file"
    echo "  Benchmark: $BENCH_NAME"
    echo "  Mode: $MODE"
    echo "----------------------------------------"

    # Skip VideoSIAH-Eval
    if [ "$BENCH_NAME" = "VideoSIAH-Eval" ]; then
        echo "  [SKIP] VideoSIAH-Eval not supported, skipping"
        echo ""
        continue
    fi

    # Check if benchmark is supported
    IS_SUPPORTED=0
    for supported in $SUPPORTED_BENCHS; do
        if [ "$BENCH_NAME" = "$supported" ]; then
            IS_SUPPORTED=1
            break
        fi
    done

    if [ "$IS_SUPPORTED" -eq 0 ]; then
        echo "  [SKIP] Unsupported benchmark: $BENCH_NAME, skipping"
        echo ""
        continue
    fi

    echo "  Execute: python eval_all.py --result_path $FULL_PATH --dataset $BENCH_NAME --mode $MODE"
    echo ""

    # Run evaluation and capture output
    EVAL_OUTPUT=$(python3 "$EVAL_SCRIPT" --result_path "$FULL_PATH" --dataset "$BENCH_NAME" --mode "$MODE" 2>&1)
    echo "$EVAL_OUTPUT"
    echo ""

    # Parse structured results from RESULT_JSON line
    RESULT_JSON_LINE=$(echo "$EVAL_OUTPUT" | grep "\[RESULT_JSON\]" | tail -1)
    ACC_VAL=""
    if [ -n "$RESULT_JSON_LINE" ]; then
        ACC_VAL=$(echo "$RESULT_JSON_LINE" | sed 's/.*"accuracy": *\([0-9.]*\).*/\1/')
    fi

    # Extract extra stats for tool mode
    EXTRA_INFO=""
    if [ "$MODE" = "tool" ]; then
        ROUNDS_LINE=$(echo "$EVAL_OUTPUT" | grep "avg_num_rounds" | tail -1)
        CALLS_LINE=$(echo "$EVAL_OUTPUT" | grep "avg_tool_calls" | tail -1)
        if [ -n "$ROUNDS_LINE" ]; then
            ROUNDS_VAL=$(echo "$ROUNDS_LINE" | sed 's/.*= *//' | sed 's/  .*//')
            EXTRA_INFO="rounds=${ROUNDS_VAL}"
        fi
        if [ -n "$CALLS_LINE" ]; then
            CALLS_VAL=$(echo "$CALLS_LINE" | sed 's/.*= *//' | sed 's/  .*//')
            if [ -n "$EXTRA_INFO" ]; then
                EXTRA_INFO="${EXTRA_INFO}, calls=${CALLS_VAL}"
            else
                EXTRA_INFO="calls=${CALLS_VAL}"
            fi
        fi
    fi

    BENCH_FULL_NAMES+=("$BENCH_NAME")
    BENCH_MODES+=("$MODE")
    BENCH_ACCURACIES+=("$ACC_VAL")
    BENCH_EXTRA+=("$EXTRA_INFO")

    COUNTER=$((COUNTER + 1))
done

# ── Summary output ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Evaluation summary: $(basename "$MODEL_DIR")"
echo "========================================"

NUM_BENCHS=${#BENCH_FULL_NAMES[@]}
if [ $NUM_BENCHS -gt 0 ]; then
    printf "  %-20s  %-10s  %s\n" "Benchmark" "Mode" "Accuracy"
    printf "  %-20s  %-10s  %s\n" "--------" "----" "--------"
    for i in $(seq 0 $((NUM_BENCHS - 1))); do
        NAME="${BENCH_FULL_NAMES[$i]}"
        MODE="${BENCH_MODES[$i]}"
        ACC="${BENCH_ACCURACIES[$i]}"
        EXTRA="${BENCH_EXTRA[$i]}"
        ACC_FMT=$(awk -v v="$ACC" 'BEGIN { printf "%.1f", v * 100 }')
        if [ -n "$EXTRA" ]; then
            printf "  %-20s  %-10s  %s  (%s)\n" "$NAME" "$MODE" "$ACC_FMT" "$EXTRA"
        else
            printf "  %-20s  %-10s  %s\n" "$NAME" "$MODE" "$ACC_FMT"
        fi
    done
else
    echo "  No valid evaluation results"
fi

echo ""
echo "[INFO] Evaluation complete, processed $COUNTER benchmark(s)"
