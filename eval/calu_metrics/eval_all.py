#!/usr/bin/env python3
"""
Unified evaluation script for all benchmarks and inference modes.

Supported evaluation modes:
  direct    Strict match: response.strip() == answer (full response is the answer letter)
  reasoning Thought mode: extract the first letter (ABCD) from <answer>...</answer> tags in response
  tool      Tool reasoning mode: same as reasoning, plus tool call statistics

Supported benchmarks: V2P-Bench, VideoMME, LongVideoBench, VideoReferSuit

Usage:
    python eval_all.py --result_path /path/to/result.jsonl --dataset V2P-Bench --mode tool
    python eval_all.py --result_path /path/to/result.jsonl --dataset VideoMME --mode reasoning
"""

import argparse
import json
from collections import defaultdict
from utils import extract_thinking_answer


# ── Dataset Config ────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "V2P-Bench": {
        "dim_field": "dimension",
        "dim_order": ["OA", "HA", "OD", "FM", "CR", "PU", "CI", "FT", "RT", "AS", "SR", "GC"],
        "dim_map": {1: "OA", 2: "HA", 3: "OD", 4: "FM", 5: "CR", 6: "PU", 7: "CI",
                    9: "FT", 10: "RT", 12: "AS", 13: "SR", 14: "GC"},
    },
    "VideoMME": {
        "dim_field": "duration",
        "dim_order": None,
        "dim_map": None,
    },
    "LongVideoBench": {
        "dim_field": "duration_group",
        "dim_order": None,
        "dim_map": None,
    },
    "VideoReferSuit": {
        "dim_field": "type",
        "dim_order": None,
        "dim_map": None,
    },
}

SUPPORTED_DATASETS = list(DATASET_CONFIG.keys())


# ── Utility Functions ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def get_correct(record: dict, mode: str):
    """Returns whether this entry is correct (True/False), or None if undetermined."""
    answer   = str(record.get("answer",   "")).strip().upper()
    response = str(record.get("response", ""))

    if not answer:
        return None

    if mode == "direct":
        pred = response.strip().upper()
        if not pred:
            return None
        return pred == answer
    else:
        pred = extract_thinking_answer(response)
        if pred is None:
            return None
        return pred == answer


def get_dim_name(record: dict, cfg: dict) -> str:
    """Extracts dimension name from the record based on dataset config."""
    dim = record.get(cfg["dim_field"])
    if cfg["dim_map"] is not None:
        return cfg["dim_map"].get(dim, str(dim)) if dim is not None else "unknown"
    return dim if dim is not None else "unknown"


def print_accuracy(results: list, mode: str, cfg: dict, dataset_name: str = "unknown"):
    total = len(results)
    error = [r for r in results if r.get("error")]
    mode_label_map = {"direct": "Strict Match", "reasoning": "Thought Mode", "tool": "Tool Mode"}
    mode_label = mode_label_map.get(mode, mode)

    pairs = []

    for r in results:
        c = get_correct(r, mode)
        if c is None:
            continue
        pairs.append((r, c))

    if not pairs:
        print("[WARN] No valid predictions to evaluate.")
        return

    correct_count = sum(int(c) for _, c in pairs)
    acc = correct_count / len(pairs)
    print(f"\n[SUMMARY] accuracy = {correct_count}/{len(pairs)} = {acc:.4f}")

    return {
        "dataset": dataset_name,
        "mode": mode,
        "correct": correct_count,
        "accuracy": acc,
    }

    # ── Per-Dimension Statistics ───────────────────────────────────────────────
    dim_correct: dict = defaultdict(int)
    dim_total: dict = defaultdict(int)
    for r, c in pairs:
        dim_name = get_dim_name(r, cfg)
        dim_total[dim_name] += 1
        dim_correct[dim_name] += int(c)

    print("\n[PER-DIMENSION]")
    dim_order = cfg["dim_order"]
    if dim_order is not None:
        for dim_name in dim_order:
            if dim_name in dim_total:
                acc_val = dim_correct[dim_name] / dim_total[dim_name] * 100
                print(f"  {dim_name:6s}: {dim_correct[dim_name]:4d}/{dim_total[dim_name]:4d} = {acc_val:.1f}%")
    else:
        for dim_name in sorted(dim_total.keys()):
            acc_val = dim_correct[dim_name] / dim_total[dim_name] * 100
            print(f"  {dim_name:30s}: {dim_correct[dim_name]:4d}/{dim_total[dim_name]:4d} = {acc_val:.1f}%")


# ── Main Entry ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for all benchmarks and inference modes"
    )
    parser.add_argument(
        "--result_path",
        required=True,
        help="Path to the inference result JSONL file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help=f"Dataset name: {', '.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "--mode",
        default="tool",
        choices=["direct", "reasoning", "tool"],
        help="Evaluation mode: direct (strict match) / reasoning (extract <answer> tag) / tool (reasoning + tool stats)",
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]

    results = load_jsonl(args.result_path)
    print(f"[INFO] Loaded result file: {args.result_path} ({len(results)} records)")
    print(f"[INFO] Dataset: {args.dataset}  Mode: {args.mode}")
    eval_result = print_accuracy(results, args.mode, cfg, args.dataset)

    if eval_result is not None:
        print(f"[RESULT_JSON] {json.dumps(eval_result)}", flush=True)


if __name__ == "__main__":
    main()
