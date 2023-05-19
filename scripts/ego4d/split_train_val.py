import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from video_blip.data.utils import C_REGEX

parser = argparse.ArgumentParser()
parser.add_argument("fho_main_path")
parser.add_argument("split_output_path")
parser.add_argument("video_dir_path")
args = parser.parse_args()

# Load fho_main.json
with open(args.fho_main_path) as f:
    data = json.load(f)

# create a dict video_uid => video
video_dir_path = Path(args.video_dir_path)
video_dict = {
    video["video_uid"]: video
    for video in data["videos"]
    # some videos in fho_main.json actually don't exist, so filter them out
    if (video_dir_path / (video["video_uid"] + ".mp4")).exists()
}

print(f"num videos before filtering: {len(video_dict)}")
# filter narrated actions
videos_to_delete = []
for video_uid, video in tqdm(video_dict.items(), desc="filtering videos"):
    video["narrated_actions"] = [
        action
        for interval in video_dict[video_uid]["annotated_intervals"]
        for action in interval["narrated_actions"]
        if not action["is_rejected"]
        and action["is_valid_action"]
        and C_REGEX.match(action["narration_text"])
    ]
    if len(video["narrated_actions"]) == 0:
        videos_to_delete.append(video_uid)
for video_uid in videos_to_delete:
    del video_dict[video_uid]
print(f"num videos after filtering: {len(video_dict)}")

# calculate the total number of filtered narrated actions
total_num_narrated_actions = sum(
    len(video["narrated_actions"]) for _, video in video_dict.items()
)
print(f"total num narrated actions: {total_num_narrated_actions}")

# following the ego4d conventions, we split 75/25.
goal_num_train_narrated_actions = round(total_num_narrated_actions * 0.75)
goal_num_val_narrated_actions = (
    total_num_narrated_actions - goal_num_train_narrated_actions
)
print(f"goal num train narrated actions: {goal_num_train_narrated_actions}")
print(f"goal num val narrated actions: {goal_num_val_narrated_actions}")

# now randomly sample videos until we hit the desired number of
# validation narrated actions.
random.seed(42)
curr_num_train_narrated_actions = 0
curr_num_val_narrated_actions = 0
# video_uid => num_narrated_actions
train_videos: dict[str, int] = {}
val_videos: dict[str, int] = {}
video_uids = list(video_dict.keys())
for video_uid in tqdm(
    random.sample(video_uids, k=len(video_uids)), desc="sampling videos"
):
    len_narrated_actions = len(video_dict[video_uid]["narrated_actions"])
    if curr_num_train_narrated_actions < goal_num_train_narrated_actions:
        train_videos[video_uid] = len(video_dict[video_uid]["narrated_actions"])
        curr_num_train_narrated_actions += len_narrated_actions
    else:
        val_videos[video_uid] = len_narrated_actions
        curr_num_val_narrated_actions += len_narrated_actions

# quick sanity check
assert curr_num_train_narrated_actions == sum(train_videos.values())
assert curr_num_val_narrated_actions == sum(val_videos.values())

print(f"sampled number of train narrated actions: {curr_num_train_narrated_actions}")
print(f"sampled number of val narrated actions: {curr_num_val_narrated_actions}")

# write to files
split_output_path = Path(args.split_output_path)
with open(split_output_path / "fho_main_train.json", "w") as f:
    json.dump({"split": "train", "videos": train_videos}, f)
with open(split_output_path / "fho_main_val.json", "w") as f:
    json.dump({"split": "val", "videos": val_videos}, f)
