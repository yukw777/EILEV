import argparse
import csv
import json

from eilev.data.ego4d import filter_action, get_structured_noun

parser = argparse.ArgumentParser()
parser.add_argument("narrated_actions_csv")
parser.add_argument("fho_main")
parser.add_argument("outfile")
args = parser.parse_args()


with open(args.fho_main) as f:
    fho_main = json.load(f)

frame_path_to_structured_verb: dict[str, str] = {}
frame_path_to_structured_noun: dict[str, str | None] = {}
for video in fho_main["videos"]:
    clip_id = 0
    for interval in video["annotated_intervals"]:
        for action in interval["narrated_actions"]:
            if not filter_action(action):
                continue
            frame_path = video["video_uid"] + "|" + str(clip_id)
            clip_id += 1
            frame_path_to_structured_verb[frame_path] = action["structured_verb"]
            frame_path_to_structured_noun[frame_path] = get_structured_noun(action)

with open(args.narrated_actions_csv, newline="") as narrated_actions_csv, open(
    args.outfile, "w", newline=""
) as outfile:
    csvreader = csv.DictReader(narrated_actions_csv)
    csvwriter = csv.DictWriter(
        outfile,
        [
            "frame_path",
            "video_uid",
            "clip_index",
            "narration_timestamp_sec",
            "narration_text",
            "structured_verb",
            "structured_noun",
        ],
    )
    csvwriter.writeheader()
    for row in csvreader:
        csvwriter.writerow(
            {
                "structured_verb": frame_path_to_structured_verb[row["frame_path"]],
                "structured_noun": frame_path_to_structured_noun[row["frame_path"]],
                **row,
            }
        )
