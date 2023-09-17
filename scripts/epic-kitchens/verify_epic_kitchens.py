import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("original_full_sent_annotation")
parser.add_argument("extracted_narrated_actions_annotation")
args = parser.parse_args()

original_annotation: dict[str, list[dict]] = defaultdict(list)

with open(args.original_full_sent_annotation, newline="") as f:
    for row in csv.DictReader(f):
        original_annotation[row["video_id"]].append(row)

extracted_annotation: dict[str, list[dict]] = defaultdict(list)
with open(args.extracted_narrated_actions_annotation, newline="") as f:
    for row in csv.DictReader(f):
        extracted_annotation[row["video_uid"]].append(row)

for video_uid in original_annotation:
    if len(original_annotation[video_uid]) != len(extracted_annotation[video_uid]):
        print(f"{video_uid}")
        i = -1
        for i in range(len(extracted_annotation[video_uid])):
            if (
                extracted_annotation[video_uid][i]["narration_text"]
                != original_annotation[video_uid][i]["full_sent_narration"]
            ):
                print(f"Difference at index {i}:")
        if i == -1 or i != len(original_annotation[video_uid]):
            print(f"Extracted shorter than original: {i}")
