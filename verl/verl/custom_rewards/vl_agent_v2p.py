import random
import re

from openai import OpenAI

# ─────────────────────────────────────────────
# LLM-as-a-Judge client
# ─────────────────────────────────────────────
client = OpenAI(
    api_key="api_key",
    base_url="base_url",
)
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"

# ─────────────────────────────────────────────
# LLM-as-a-Judge prompt helpers
# ─────────────────────────────────────────────

def _get_chat_template() -> str:
    return """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question,
and [Model_answer] is the answer extracted from a model's output to this question.

Judge how consistent the two answers are.

Scoring rules
• 1    — Fully consistent: they convey the same meaning (e.g., "pink" vs. "it is pink").
• 0.5 — Partially consistent: they overlap on some key points but not all.
• 0    — Inconsistent: they conflict or share no essential overlap.

Output **only** one of the following numbers: 1, 0.5, or 0.
"""


def _get_few_shot_examples() -> list[str]:
    return [
        """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
        """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""",
        """
[Question]: What happens immediately after the fireworks illuminate the sky?
[Standard Answer]: The crowd cheers loudly and waves flags.
[Model_answer] : The crowd cheers.
Judgement: 0.5
""",
        """
[Question]: What items does the waitress hand to the customer?
[Standard Answer]: She hands over a sandwich and a cup of coffee.
[Model_answer] : She hands over a sandwich and a cup of tea.
Judgement: 0.5
""",
        """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""",
        """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""",
    ]


def _build_llm_judge_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    prompt = _get_chat_template()
    for example in _get_few_shot_examples():
        prompt += example + "\n"
    prompt += f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    return prompt


def _call_llm_judge(answer_text: str, ground_truth: str, question: str) -> float:
    """Call the LLM judge and return acc_reward in {0.0, 0.5, 1.0}.
    Falls back to rule-based accuracy if the API call fails after 5 attempts."""
    full_prompt = _build_llm_judge_prompt(answer_text, ground_truth, question)

    for attempt in range(5):
        try:
            chat_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1_000_000),
                temperature=0.3,
            )
            response = chat_response.choices[0].message.content
            if response is None:
                if attempt < 4:
                    continue
                print("All 5 LLM judge attempts failed, falling back to rule-based acc.")
                return _rule_acc(answer_text, ground_truth)
            response = response.strip()
            if attempt < 4:
                print(f" [compute_score] LLM judge response: {response}")
        except Exception as e:
            if attempt < 4:
                continue
            print("All 5 LLM judge attempts failed, falling back to rule-based acc.")
            return _rule_acc(answer_text, ground_truth)

        # strip prefix "Judgement: " if present
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                return 1.0
            if "0.5" in response:
                return 0.5
            if "0" in response:
                return 0.0
            if attempt < 4:
                continue
            print("All 5 LLM judge attempts failed, falling back to rule-based acc.")
            return _rule_acc(answer_text, ground_truth)
        else:
            if response == "1":
                return 1.0
            if response == "0.5":
                return 0.5
            if response == "0":
                return 0.0
            if attempt < 4:
                continue
            print("All 5 LLM judge attempts failed, falling back to rule-based acc.")
            return _rule_acc(answer_text, ground_truth)

    # should not reach here, but just in case
    print("All 5 LLM judge attempts failed, falling back to rule-based acc.")
    return _rule_acc(answer_text, ground_truth)


# ─────────────────────────────────────────────
# Format check
# ─────────────────────────────────────────────

def _check_format(predict_str: str) -> tuple[bool, str]:
    """
    Check whether the prediction strictly follows the required format.

    Expected patterns:
    - No tool:   <think>...</think><answer>...</answer>
    - With tool: <think>...</think><tool_call>...</tool_call><think>...</think>...<answer>...</answer>

    Returns:
        (is_format_error, answer_text)
    """
    is_format_error = False

    count_think_open = predict_str.count("<think>")
    count_think_close = predict_str.count("</think>")
    count_tool_open = predict_str.count("<tool_call>")
    count_tool_close = predict_str.count("</tool_call>")
    count_answer_open = predict_str.count("<answer>")
    count_answer_close = predict_str.count("</answer>")

    # basic tag pairing
    if count_think_open != count_think_close or count_think_open == 0:
        is_format_error = True
    if count_tool_open != count_tool_close:
        is_format_error = True
    if count_answer_open != count_answer_close or count_answer_open != 1:
        is_format_error = True

    # strict structural check
    if not is_format_error:
        stripped = predict_str.strip()
        if count_tool_open == 0:
            # no tool case
            pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            if not re.match(pattern, stripped, re.DOTALL):
                is_format_error = True
        else:
            # tool call case: must start with <think> and end with </answer>
            if not (stripped.startswith("<think>") and stripped.endswith("</answer>")):
                is_format_error = True
            else:
                # verify alternating tag sequence: think (tool_call think)* answer
                tags = re.findall(r"<(think|tool_call|answer)>", stripped)
                expected = ["think"]
                for _ in range(count_tool_open):
                    expected.extend(["tool_call", "think"])
                expected.append("answer")
                if tags != expected:
                    is_format_error = True

    # extract answer text
    if count_answer_open == 0 or count_answer_close == 0:
        answer_text = ""
    else:
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # penalize extremely long answers (anti-hack)
    if len(answer_text) >= 1000:
        is_format_error = True
        answer_text = ""

    return is_format_error, answer_text


# ─────────────────────────────────────────────
# Rule-based accuracy
# ─────────────────────────────────────────────
def _rule_acc(answer_text: str, ground_truth: str) -> float:
    """
    Simple rule-based exact/relaxed match.
    Returns 1.0 if the prediction matches the ground truth, else 0.0.
    """
    if not answer_text:
        return 0.0

    pred = answer_text.strip().rstrip(".").lower()
    gt = ground_truth.strip().rstrip(".").lower()

    if not pred:
        return 0.0

    # exact match by first char (MCQ: "A: ..." → 'a' == 'a')
    if pred[0] == gt:
        return 1.0

    # relaxed match: one contains the other with ≥ 0.8 length ratio
    relax = 0.8
    if pred in gt and len(pred) >= relax * len(gt):
        return 1.0
    if gt in pred and len(gt) >= relax * len(pred):
        return 1.0

    return 0.0


# ─────────────────────────────────────────────
# Per-dataset compute_score implementations
# ─────────────────────────────────────────────

def compute_score_v2p_bench_rule(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> tuple[float, float, float]:
    """
    V2P-Bench: rule-based accuracy + format reward.
    Score = 0.8 * acc + 0.2 * format
    """
    is_format_error, answer_text = _check_format(predict_str)
    format_reward = 0.0 if is_format_error else 1.0
    acc_reward = _rule_acc(answer_text, ground_truth)
    # acc_reward = _rule_acc(answer_text, ground_truth) if not is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward
    return score, acc_reward, format_reward


def compute_score_v2p_bench_llm(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> tuple[float, float, float]:
    """
    V2P-Bench: LLM-as-a-Judge accuracy + format reward.
    Score = 0.8 * acc + 0.2 * format
    """
    is_format_error, answer_text = _check_format(predict_str)
    format_reward = 0.0 if is_format_error else 1.0

    if not answer_text:
        acc_reward = 0.0
    else:
        question_text = (extra_info or {}).get("question", "")
        acc_reward = _call_llm_judge(answer_text, ground_truth, question_text)

    score = 0.8 * acc_reward + 0.2 * format_reward
    return score, acc_reward, format_reward


def compute_score_temp1(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> tuple[float, float, float]:
    """Placeholder for temp1 dataset."""
    is_format_error, answer_text = _check_format(predict_str)
    format_reward = 0.0 if is_format_error else 1.0
    acc_reward = _rule_acc(answer_text, ground_truth) if not is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward
    return score, acc_reward, format_reward


def compute_score_temp2(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> tuple[float, float, float]:
    """Placeholder for temp2 dataset."""
    is_format_error, answer_text = _check_format(predict_str)
    format_reward = 0.0 if is_format_error else 1.0
    acc_reward = _rule_acc(answer_text, ground_truth) if not is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward
    return score, acc_reward, format_reward
