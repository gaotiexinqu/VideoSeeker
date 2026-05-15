#!/usr/bin/env python3
"""
LongVT Evaluation Script - Adapted from lmms-eval
Uses vLLM-deployed Judge model for evaluation.

Reference:
- lmms_eval/tasks/_task_utils/reasoning_utils.py
- lmms_eval/tasks/longvt/utils.py
"""

import json
import os
import re
import sys
import requests
from typing import Optional, Dict, Any, List
from openai import OpenAI


# ============================================================
# Judge Model Configuration
# ============================================================
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", "http://localhost:1234/v1")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "judge")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "EMPTY")
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "True")

# Initialize OpenAI client
client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_API_BASE,
    timeout=120  # Increase timeout to 120 seconds
)

JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Input
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0"""

JUDGE_PROMPT_WITH_ANSWER = """
You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case. You will receive the question, the ground truth answer, and the model prediction.

# Input
Question:
```
{question}
```

Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0
"""

def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the boxed answer.

    Returns:
        str: The extracted answer from \boxed{}, or an empty string if not found.
    """
    # Find all occurrences of \boxed{
    boxed_start = "\\boxed{"
    start_indices = []

    # Find all positions where \boxed{ starts
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1

    if not start_indices:
        return ""

    # For each \boxed{ occurrence, find the matching closing brace
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1  # Position at the opening brace of \boxed{

        while pos < len(predict_str):
            char = predict_str[pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1

    # Return the last (rightmost) match if multiple found
    return results[-1] if results else ""


def extract_anwser_tag(predict_str: str) -> str:
    """Extract the answer tag from the prediction string.

    This function now handles both <answer> tags and \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the answer tag.

    Returns:
        str: The extracted answer tag, or an empty string if not found.
    """
    # First try to extract from <answer> tags
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)

    # If no <answer> tag found, try to extract from \boxed{} format
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer

    # If neither format found, try to extract the last number or expression
    # This is a fallback for cases where the answer is just stated without formatting
    lines = predict_str.strip().split("\n")
    for line in reversed(lines):
        # Look for patterns like "The answer is 204" or just "204"
        if line.strip():
            # Try to find numbers at the end of the line
            number_match = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
            if number_match:
                return number_match.group(1)

    return ""

def format_reward(predict_str: str) -> float:
    """Check if the prediction string follows the expected format.

    Now handles both <think><answer> format and \boxed{} format.
    """
    # Check for <think>.*</think>.*<answer>.*</answer> pattern
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0

    analysis_answer_pattern = re.compile(r"<analysis>.*</analysis>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(analysis_answer_pattern, predict_str):
        return 1.0

    # Check for \boxed{} format (common in mathematical solutions)
    if extract_boxed_answer(predict_str):
        return 1.0

    # Check for basic answer format (contains some mathematical content and ends with a number)
    if len(predict_str.strip()) > 50:  # Reasonable solution length
        # Look for mathematical expressions or reasoning
        has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
        # Look for final answer
        has_answer = bool(extract_anwser_tag(predict_str))

        if has_math and has_answer:
            return 0.8  # Partial credit for reasonable format

    return 0.0


def simple_parse(predict_str: str) -> str:
    """Parse the prediction string to extract the answer.

    Args:
        predict_str (str): The prediction string to be parsed.

    Returns:
        str: The parsed answer from the prediction string.
    """
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]

    return predict_str.strip()


def parse_mcq(predict_str: str) -> str:
    """
    Parse multiple choice answers from various formats.
    Handles formats like: "A", "A.", "A)", "(A)", "The answer is A", "A: xxx", etc.
    """
    if not predict_str or predict_str.strip() == "":
        return ""

    # Clean up the response
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)

    # Add spaces to avoid partial matches
    response = " " + response + " "

    # All possible choice letters (extend if needed)
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]

    candidates = []

    # Pattern 1: Look for choices with parentheses e.g., (A), (B), (C), (D)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))

    # Pattern 2: Look for choices with periods e.g., A., B., C., D.
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))

    # Pattern 3: Look for choices with colons e.g., A:, B:, C:, D:
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))

    # Pattern 4: Look for choices with right parentheses e.g., A), B), C), D)
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))

    # Pattern 5: Look for choices with spaces after e.g., A B C D
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))

    # Pattern 6: Look for choices with dashes e.g., A- B- C- D-
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))

    # Pattern 7: Look for choices with underscores e.g., A_ B_ C_ D_
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))

    # Pattern 8: Look for choices with equal signs e.g., A= B= C= D=
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))

    # Pattern 9: Look for common answer phrases followed by choices
    answer_phrases = [
        "the answer is",
        "answer is",
        "the correct answer is",
        "correct answer is",
        "the answer",
        "answer",
        "correct answer",
        "the correct answer",
        "the best answer is",
        "best answer is",
        "the best answer",
        "best answer",
        "the option is",
        "option is",
        "the correct option is",
        "correct option is",
        "the choice is",
        "choice is",
        "the correct choice is",
        "correct choice is",
        "i choose",
        "i select",
        "i pick",
        "my answer is",
        "my choice is",
    ]

    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            # Look for choices after the phrase
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))

    # Pattern 10: Look for choices at the very beginning of the response
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))

    # Pattern 11: Look for choices at the very end of the response
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    # Pattern 12: Look for choices with numbers (e.g., "1. A", "2. B")
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))

    # If no candidates found, try to extract from the entire response
    if not candidates:
        # Look for any choice letter in the response
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    # Return the best candidate
    if candidates:
        # Sort by position (later in text) and priority of format
        format_priority = {"start": 10, "end": 9, "numbered": 8, "phrase": 7, "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3, "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0}

        # Sort by format priority first, then by position
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]

    return ""


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """Check if the prediction string matches the ground truth exactly.

    Args:
        predict_str (str): The prediction string to be checked.
        ground_truth (str): The ground truth string for comparison.
        relax_portion (float): The minimum portion of length required for partial matches.

    Returns:
        float: 1.0 if the prediction matches the ground truth, otherwise 0.0.
    """
    # If the question is an mcq
    if parse_mcq(ground_truth) in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = parse_mcq(predict_str)
        if predict_str.lower().strip() == parse_mcq(ground_truth).lower().strip():
            return 1.0
        return 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def llm_as_judge_sync(predict_str, ground_truth, extra_info):
    if extra_info is not None and "question" in extra_info:
        prompt = JUDGE_PROMPT_WITH_ANSWER.format(question=extra_info["question"], answer=ground_truth, prediction=predict_str)
    else:
        prompt = JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "max_tokens": 5,
        "model": JUDGE_MODEL_NAME,
    }
    response = client.chat.completions.create(**payload)
    try:
        score = int(response.choices[0].message.content)
    except Exception:
        score = 0
    return score

def acc_reward(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    format_reward_score: float = 0.0,
    solution_str: Optional[str] = None
) -> float:
    """Compute the accuracy reward for a given prediction.

    Uses multiple strategies in order:
    1. relax_exact_match
    2. llm_as_judge_sync (if USE_LLM_JUDGE=True)

    Args:
        predict_str: The extracted prediction.
        ground_truth: The ground truth answer.
        extra_info: Additional information for judge.
        format_reward_score: Format reward score for fallback logic.
        solution_str: Original solution string for fallback judgment.

    Returns:
        Accuracy score (0.0 or 1.0).
    """
    predict_str = simple_parse(predict_str)
    gt = simple_parse(ground_truth)

    # Strategy 1: Relaxed exact match
    acc_score = relax_exact_match(predict_str, gt)
    if acc_score == 1.0:
        return 1.0

    # Strategy 2: LLM Judge (if enabled and exact match failed)
    if USE_LLM_JUDGE == "True":
        acc_score = llm_as_judge_sync(predict_str, ground_truth, extra_info)
        if acc_score == 1.0:
            return 1.0

        # Try with original solution string if length is short
        if solution_str is not None and len(solution_str) < 4000:
            acc_score = llm_as_judge_sync(solution_str, ground_truth, extra_info)
            return 1.0 if acc_score == 1 else 0.0

    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """Compute the score for a given solution.

    Combines accuracy score and format reward score.

    Args:
        solution_str: The solution string to evaluate.
        ground_truth: The ground truth answer.
        extra_info: Additional information for evaluation.

    Returns:
        Dictionary containing score, acc_score, format_reward_score, etc.
    """

    format_score = 0.2  # Weight for format reward
    format_reward_score = format_reward(solution_str)

    # Extract answer from tags
    extracted_answer = extract_anwser_tag(solution_str).strip()

    # Calculate accuracy score
    acc_score = acc_reward(
        extracted_answer,
        ground_truth,
        extra_info,
        format_reward_score,
        solution_str
    )

    predict_str = simple_parse(extracted_answer)
    gt = simple_parse(ground_truth)

    # Final score = 90% accuracy + 10% format
    score = (1.0 - format_score) * acc_score + format_score * format_reward_score

    return {
        "score": score,
        "acc_score": acc_score,
        "format_reward_score": format_reward_score,
        "predict_str": predict_str,
        "ground_truth": gt,
        "extracted_answer": extracted_answer
    }

def longvt_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process results for LongVT evaluation.

    Args:
        doc: Document containing question and answer.
        results: List of predictions.

    Returns:
        Dictionary with acc_score and format_score.
    """
    question = doc.get("question", "")
    answer = doc.get("answer", "")
    extra_info = {"question": question}

    acc_score = 0.0
    format_score = 0.0

    for pred in results:
        score_dict = compute_score(
            solution_str=pred.strip(),
            ground_truth=answer,
            extra_info=extra_info
        )
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    n = len(results) if results else 1
    return {
        "acc_score": acc_score / n,
        "format_score": format_score / n
    }

def evaluate_jsonl(
    input_file: str,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate predictions in a JSONL file.

    Args:
        input_file: Path to input JSONL file.
        output_file: Optional path to save detailed results.
        verbose: Whether to print progress.

    Returns:
        Dictionary with aggregate metrics.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    results = []
    total_acc_score = 0.0
    total_format_score = 0.0
    total_count = 0

    print(f"\n{'='*60}")
    print(f"Starting evaluation: {input_file}")
    print(f"Judge API: {JUDGE_API_BASE}")
    print(f"Judge Model: {JUDGE_MODEL_NAME}")
    print(f"LLM Judge: {USE_LLM_JUDGE}")
    print(f"{'='*60}\n")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            data = json.loads(line)

            if "raw_output" in data:
                solution_str = data["raw_output"]
            else:
                print(f"[Warning] Line {line_num}: Missing raw_output field")
                continue

            ground_truth = data.get("answer", "")
            question = data.get("question", "")
            video_id = data.get("video_id", data.get("id", f"line_{line_num}"))

            # Calculate score
            extra_info = {"question": question}
            score_dict = compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info
            )

            # Update statistics
            total_acc_score += score_dict["acc_score"]
            total_format_score += score_dict.get("format_reward_score", 0.0)
            total_count += 1

            # Record result
            num_tool_calls = len(data.get("tool_calls", []))
            result_item = {
                "id": video_id,
                "question": question,
                "ground_truth": ground_truth,
                "predict": data.get("predict", ""),
                "extracted_answer": score_dict["extracted_answer"],
                "acc_score": score_dict["acc_score"],
                "format_score": score_dict.get("format_reward_score", 0.0),
                "score": score_dict["score"],
                "num_tool_calls": num_tool_calls,
            }
            results.append(result_item)

            if verbose and (line_num <= 10 or line_num % 100 == 0):
                print(f"  [{line_num}] {video_id}: acc={score_dict['acc_score']:.2f}, "
                      f"format={score_dict.get('format_reward_score', 0.0):.2f}, "
                      f"GT={ground_truth}, Pred={score_dict['extracted_answer'][:50]}")

    # Calculate average score
    if total_count > 0:
        avg_acc_score = total_acc_score / total_count
        avg_format_score = total_format_score / total_count

        avg_score = 0.8 * avg_acc_score + 0.2 * avg_format_score
    else:
        avg_acc_score = 0.0
        avg_format_score = 0.0
        avg_score = 0.0

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Total {total_count} samples")
    print(f"{'='*60}")
    print(f"  Accuracy (acc_score):  {avg_acc_score:.4f}")
    print(f"  Format Score (format_score): {avg_format_score:.4f}")
    print(f"  Overall Score (score):      {avg_score:.4f}")
    print(f"{'='*60}\n")

    # Save detailed results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            # Append summary line
            summary_line = {
                "type": "summary",
                "total_count": total_count,
                "acc_score": avg_acc_score,
                "format_score": avg_format_score,
                "score": avg_score
            }
            f.write(json.dumps(summary_line, ensure_ascii=False) + "\n")
        print(f"Detailed results saved to: {output_file}")

    return {
        "acc_score": avg_acc_score,
        "format_score": avg_format_score,
        "score": avg_score,
        "total_count": total_count
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LongVT Evaluation Script (Using LLM Judge)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output detailed results JSON file path"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Judge API base URL (default: http://localhost:1234/v1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Judge model name (default: judge)"
    )
    parser.add_argument(
        "--no-verbose", "-q",
        action="store_true",
        help="Silent mode, do not print detailed progress"
    )

    args = parser.parse_args()

    # Update configuration
    if args.api_base:
        global JUDGE_API_BASE
        JUDGE_API_BASE = args.api_base
        # Recreate client
        global client
        client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_API_BASE, timeout=60)

    if args.model:
        global JUDGE_MODEL_NAME
        JUDGE_MODEL_NAME = args.model

    # Run evaluation
    result = evaluate_jsonl(
        input_file=args.input,
        output_file=args.output,
        verbose=not args.no_verbose
    )

    return result


if __name__ == "__main__":
    main()
