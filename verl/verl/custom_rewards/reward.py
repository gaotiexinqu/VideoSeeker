"""
Reward entry point for V2P-related datasets.

Dataset routing:
  - "V2P-Bench"        → rule-based accuracy  (default)
  - "V2P-Bench-llm"   → LLM-as-a-Judge accuracy
  - "temp1"           → placeholder (rule-based)
  - "temp2"           → placeholder (rule-based)

Score formula: 0.8 * acc + 0.2 * format
"""

from custom_rewards_v2p import vl_agent_v2p

# Called by class NaiveRewardManager
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    **kwargs,
) -> dict:
    """
    Compute reward score for a given solution.

    Args:
        data_source:   Dataset identifier (e.g. "V2P-Bench").
        solution_str:  Full model output string.
        ground_truth:  Reference answer string.
        extra_info:    Additional metadata dict (must contain "question" for LLM judge).

    Returns:
        dict with keys:
            score              – final weighted score
            acc_score          – accuracy component
            format_reward_score – format component
            predict_str        – original model output (for logging)
            ground_truth       – reference answer (for logging)
    """
    score_dict = {}

    # ── rule-based (default) ───────────────────────────────────
    if data_source in ("LLaVA-Video"):
        score, acc_score, format_reward_score = vl_agent_v2p.compute_score_v2p_bench_rule(
            solution_str, ground_truth, extra_info, **kwargs
        )

    # ── LLM-as-a-Judge ─────────────────────────────────────────
    elif data_source in ("LLaVA-Video-OE"):
        score, acc_score, format_reward_score = vl_agent_v2p.compute_score_v2p_bench_llm(
            solution_str, ground_truth, extra_info, **kwargs
        )

    # ── Unknown dataset: fall back to rule-based ───────────────────────────
    else:
        print(f"[WARNING] Unknown data_source '{data_source}', falling back to rule-based reward.")
        score, acc_score, format_reward_score = vl_agent_v2p.compute_score_v2p_bench_rule(
            solution_str, ground_truth, extra_info, **kwargs
        )

    score_dict["score"] = score
    score_dict["acc_score"] = acc_score
    score_dict["format_reward_score"] = format_reward_score
    score_dict["predict_str"] = solution_str
    score_dict["ground_truth"] = ground_truth

    # return NaiveRewardManager
    return score_dict
