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
from torchvision.transforms import Compose, Resize
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import Blip2Config

from eilev.data.ego4d import Ego4dFHOMainDataset

parser = argparse.ArgumentParser()
parser.add_argument("--fho_main_path", required=True)
parser.add_argument("--split_path", required=True)
parser.add_argument("--video_dir", required=True)
parser.add_argument("--frames_dir", required=True)
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--num_subsample_frames", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--max_num_narrated_actions", type=int, default=0)
parser.add_argument("--csv_only", action="store_true")
args = parser.parse_args()


def extract_frames(pixel_values: torch.Tensor, frame_path: str) -> str:
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


config = Blip2Config.from_pretrained(args.model_name_or_path)

dataset = Ego4dFHOMainDataset(
    args.fho_main_path,
    args.split_path,
    args.video_dir,
    transform=partial(
        transform,
        Compose(
            [
                UniformTemporalSubsample(args.num_subsample_frames),
                Resize(
                    (
                        # we resize to 2x of the vision model image size
                        # since we will be using RandomResizedCrop with
                        # min_scale=0.5 and max_scale=2.0
                        config.vision_config.image_size * 2,
                        config.vision_config.image_size * 2,
                    ),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        ),
    ),
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
            "structured_verb",
            "structured_noun",
        ],
    )

    # Write header row
    csv_writer.writeheader()

    num_extracted_narrated_action = 0
    for item in tqdm(
        DataLoader(dataset, batch_size=None, num_workers=args.num_workers),
        desc="Extracting frames",
    ):
        frame_path = item["video_uid"] + "|" + str(item["clip_index"])
        if not args.csv_only:
            extract_frames(item["pixel_values"], frame_path)
        csv_writer.writerow(
            {
                "frame_path": frame_path,
                "video_uid": item["video_uid"],
                "clip_index": item["clip_index"],
                "narration_timestamp_sec": item["narration_timestamp_sec"],
                "narration_text": item["narration_text"].strip(),
                "structured_verb": item["structured_verb"],
                "structured_noun": item["structured_noun"],
            }
        )
        num_extracted_narrated_action += 1
        if (
            args.max_num_narrated_actions > 0
            and num_extracted_narrated_action == args.max_num_narrated_actions
        ):
            break
