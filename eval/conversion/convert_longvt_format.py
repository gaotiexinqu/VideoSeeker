import json

input_path = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/VideoSIAH-Eval/data/test-00000-of-00001.json"
output_path = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/VideoSIAH-Eval/data/test-00000-of-00001.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item["frame_path"] = ""

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. {len(data)} records processed, frame_path field added.")
