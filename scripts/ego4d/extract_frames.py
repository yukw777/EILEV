import argparse
import csv
import os
from collections.abc import Callable
from functools import partial
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import DataLoader
from tqdm import tqdm

from video_blip.data.ego4d import Ego4dFHOMainDataset

parser = argparse.ArgumentParser()
parser.add_argument("--fho_main_path", required=True)
parser.add_argument("--split_path", required=True)
parser.add_argument("--video_dir", required=True)
parser.add_argument("--frames_dir", required=True)
parser.add_argument("--num_subsample_frames", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--max_num_narrated_actions", type=int, default=0)
args = parser.parse_args()


def process_narrated_action(
    pixel_values: torch.Tensor, video_uid: str, clip_index: int
) -> str:
    frame_path = video_uid + "|" + str(clip_index)

    # Create a dir for the extracted frames
    frames_dir = os.path.join(args.frames_dir, frame_path)
    os.makedirs(frames_dir, exist_ok=True)

    for i, frame in enumerate(
        pixel_values.permute(1, 2, 3, 0).numpy().astype(np.uint8)
    ):
        iio.imwrite(
            os.path.join(frames_dir, frame_path + "|" + str(i) + ".png"),
            frame,
            extension=".png",
        )
    return frame_path


def transform(
    video_transform: Callable[[torch.Tensor], torch.Tensor],
    item: dict[str, Any],
) -> dict[str, torch.Tensor]:
    pixel_values = item.pop("video")
    pixel_values = video_transform(pixel_values)
    return {"pixel_values": pixel_values, **item}


dataset = Ego4dFHOMainDataset(
    args.fho_main_path,
    args.split_path,
    args.video_dir,
    transform=partial(transform, UniformTemporalSubsample(args.num_subsample_frames)),
    random_clip=False,
)

# Create a directory to save all the results
os.makedirs(args.frames_dir, exist_ok=True)

# Open narrated_actions.csv file for writing
with open(
    os.path.join(args.frames_dir, "narrated_actions.csv"), "w", newline=""
) as csvfile:
    # Initialize CSV writer
    csv_writer = csv.DictWriter(
        csvfile,
        [
            "frame_path",
            "video_uid",
            "clip_index",
            "narration_timestamp_sec",
            "narration_text",
        ],
    )

    # Write header row
    csv_writer.writeheader()

    num_extracted_narrated_action = 0
    for item in tqdm(
        DataLoader(dataset, batch_size=None, num_workers=args.num_workers),
        desc="Extracting frames",
    ):
        frame_path = process_narrated_action(
            item["pixel_values"], item["video_uid"], item["clip_index"]
        )
        csv_writer.writerow(
            {
                "frame_path": frame_path,
                "video_uid": item["video_uid"],
                "clip_index": item["clip_index"],
                "narration_timestamp_sec": item["narration_timestamp_sec"],
                "narration_text": item["narration_text"].strip(),
            }
        )
        num_extracted_narrated_action += 1
        if (
            args.max_num_narrated_actions > 0
            and num_extracted_narrated_action == args.max_num_narrated_actions
        ):
            break
