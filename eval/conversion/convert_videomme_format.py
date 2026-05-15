import argparse
import ast
import csv
import json
import os
import re

parser = argparse.ArgumentParser(description='Convert Video-MME TSV to JSON')
parser.add_argument('--tsv_path',
                    default='/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/Video-MME.tsv')
parser.add_argument('--save_path',
                    default='/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme.json')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

records = []
with open(args.tsv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        video_id = row['video'].strip()
        candidates_raw = row['candidates'].strip()
        try:
            candidates = ast.literal_eval(candidates_raw)
            if isinstance(candidates, list):
                candidates = [re.sub(r"^[A-D]\.\s*", "", str(c)).strip() for c in candidates]
            else:
                candidates = []
        except Exception:
            candidates = []

        question_text = row['question'].strip()
        if candidates:
            options_text = '\n'.join([f"{chr(65+i)}: {c}" for i, c in enumerate(candidates)])
            question_text = f"{question_text}\n{options_text}"

        answer = row['answer'].strip().upper()

        record = {
            "video_path": f"{video_id}.mp4",
            "question": question_text,
            "type": row['task_type'].strip(),
            "answer": answer,
            "domain": row['domain'].strip(),
            "sub_category": row['sub_category'].strip(),
            "duration": row['duration'].strip(),
            "subtitle_path": row['subtitle_path'].strip(),
        }
        records.append(record)

with open(args.save_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"[INFO] Conversion complete. Total: {len(records)}")
print(f"[INFO] Save path: {args.save_path}")
print("\n[EXAMPLE] First record:")
print(json.dumps(records[0], ensure_ascii=False, indent=2))
