import csv
import os
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from pytorchvideo.data import LabeledVideoDataset

from eilev.data.utils import NarratedActionClipSampler, parse_timestamp


class EpicKitchensDataset(LabeledVideoDataset):
    ONE_HUNDRED_REGEX = re.compile(r"P\d\d_1\d\d")

    def __init__(
        self,
        annotation_path: str,
        epic_kitchen_55_video_dir_path: str,
        epic_kitchen_100_video_dir_path: str,
        transform: Callable[[dict], Any] | None = None,
        random_clip: bool = False,
    ) -> None:
        """
        :param annotation_path: path to the annotation file with full sentence
            narrations
        :param epic_kitchen_55_video_dir_path: path to epic kitchens 55 video dir
        :param epic_kitchen_100_video_dir_path: path to epic kitchens 100 video dir
        :param transform: optional transform function
        :param random_clip: whether to sample clips randomly
        """
        self.annotation_path = annotation_path
        self.epic_kitchens_55_video_dir_path = epic_kitchen_55_video_dir_path
        self.epic_kitchens_100_video_dir_path = epic_kitchen_100_video_dir_path

        # map from video_id => [narration_data, ...]
        video_dict: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(annotation_path) as ann_f:
            for row in csv.DictReader(ann_f):
                # translate timestamps into floats so NarratedActionClipSampler
                # can handle them
                if row["narration_timestamp"]:
                    row["narration_timestamp_sec"] = parse_timestamp(
                        row["narration_timestamp"]
                    )
                else:
                    # some actions don't have narration timestamps
                    # in that case, just take the middle of the clip.
                    row["narration_timestamp_sec"] = (
                        parse_timestamp(row["start_timestamp"])
                        + parse_timestamp(row["stop_timestamp"])
                    ) / 2
                video_dict[row["video_id"]].append(row)

        labeled_video_paths: list[tuple[str, dict]] = []
        for video_id, narration_data in video_dict.items():
            participant_id = video_id.split("_")[0]
            if self.ONE_HUNDRED_REGEX.match(video_id):
                # EPIC-KITCHENS-100 video
                video_path = os.path.join(
                    self.epic_kitchens_100_video_dir_path,
                    participant_id,
                    "videos",
                    video_id + ".MP4",
                )
            else:
                # EPIC-KITCHENS-55 video
                # The EPIC-KITCHENS-100 annotation doesn't follow the original splits
                # from EPIC-KITCHENS-55, so the video may be in the "train" directory
                # or "test" directory.
                # First check if it's in "train"
                video_path = os.path.join(
                    self.epic_kitchens_55_video_dir_path,
                    "videos/train",
                    participant_id,
                    video_id + ".MP4",
                )
                if not os.path.exists(video_path):
                    # now check if it's in "test"
                    video_path = os.path.join(
                        self.epic_kitchens_55_video_dir_path,
                        "videos/test",
                        participant_id,
                        video_id + ".MP4",
                    )
                    if not os.path.exists(video_path):
                        # we can't find the video so raise an exception
                        raise Exception(f"Video file {video_id}.MP4 not found.")

            labeled_video_paths.append(
                (video_path, {"narrated_actions": narration_data})
            )

        self.num_narrations = sum(
            len(narration_data["narrated_actions"])
            for _, narration_data in labeled_video_paths
        )

        def _transform(item: dict) -> Any:
            """The first transform function that formats `narrated_actions` and
            `video`."""
            # format narrated_actions
            narrated_actions = item.pop("narrated_actions")
            item.update(narrated_actions[item["clip_index"]])

            # turn video tensor to torch.uint8
            item["video"] = item["video"].to(torch.uint8)
            if transform is not None:
                item = transform(item)
            return item

        super().__init__(
            labeled_video_paths,
            NarratedActionClipSampler(random_clip),
            transform=_transform,
            decode_audio=False,
        )

    def __len__(self) -> int:
        return self.num_narrations
