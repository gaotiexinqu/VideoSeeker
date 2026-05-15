import json
import os
import ast

input_path = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/LongVideoBench.json"
output_path = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/LongVideoBench_converted.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def parse_candidates(candidates):
    """Parse candidates into a list of options."""
    if isinstance(candidates, list):
        return candidates
    elif isinstance(candidates, str):
        try:
            return ast.literal_eval(candidates)
        except (ValueError, SyntaxError):
            pass
        candidates = candidates.strip()
        if candidates.startswith("["):
            candidates = candidates[1:-1]
        return [c.strip().strip("'\"") for c in candidates.split(",") if c.strip()]
    return []


def get_option_letter(correct_choice):
    """Convert correct_choice to option letter."""
    choice_map = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E"}
    return choice_map.get(str(correct_choice), "A")


converted_data = []
for item in data:
    question_raw = item["question"]
    candidates_str = item["candidates"]
    correct_choice = item["correct_choice"]

    candidates_list = parse_candidates(candidates_str)

    option_letters = ["A", "B", "C", "D", "E"]
    options_text = []
    for i, cand in enumerate(candidates_list):
        if i < len(option_letters):
            opt_letter = option_letters[i]
        else:
            opt_letter = chr(ord('A') + i)
        options_text.append(f"{opt_letter}: {cand}")

    new_question = question_raw + "\n" + "\n".join(options_text)

    answer = get_option_letter(correct_choice)

    new_item = {
        "video_path": os.path.basename(item["video_path"]),
        "frame_path": "",
        "question": new_question,
        "question_raw": question_raw,
        "answer": answer,
        "video_id": item["video_id"],
        "question_wo_referring_query": item["question_wo_referring_query"],
        "candidates": candidates_list,
        "correct_choice": item["correct_choice"],
        "position": item["position"],
        "topic_category": item["topic_category"],
        "question_category": item["question_category"],
        "level": item["level"],
        "id": item["id"],
        "subtitle_path": item["subtitle_path"],
        "duration_group": item["duration_group"],
        "starting_timestamp_for_subtitles": item["starting_timestamp_for_subtitles"],
        "duration": item["duration"],
        "view_count": item["view_count"],
        "index": item["index"],
        "video": item["video"],
    }
    converted_data.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"Conversion complete: {output_path}")
print(f"Total: {len(converted_data)} records")
