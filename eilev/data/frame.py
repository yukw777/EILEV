import json
import random
from collections import defaultdict
from collections.abc import Callable
from csv import DictReader
from pathlib import Path
from typing import Any

import torch
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import Dataset


class FrameDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frames_dir: str,
        annotation_file: str | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        data_filter: Callable[[dict[str, Any]], bool] | None = None,
        return_frames: bool = True,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param annotation_file: path to annotation file. If frames_dir contains
            narrated_actions.csv, this is optional.
        :param transform: transform function to be called for each datapoint
        :param data_filter: function to be used to filter datapoints
        :param return_frames: whether to return frame data for each datapoint or not
        """
        self.frames_dir = Path(frames_dir)
        self.return_frames = return_frames
        self.data: list[dict] = []
        self.dict_data: dict[str, dict] = {}
        if annotation_file is None:
            self.annotation_file_path = self.frames_dir / "narrated_actions.csv"
        else:
            self.annotation_file_path = Path(annotation_file)
        assert self.annotation_file_path.exists()
        with open(self.annotation_file_path, newline="") as csvfile:
            csvreader = DictReader(csvfile)
            for row in csvreader:
                if data_filter is not None and not data_filter(row):
                    continue
                self.data.append(row)
                self.dict_data[row["frame_path"]] = row

        self._video_path_handler = VideoPathHandler()
        self._transform = transform

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        if isinstance(index, int):
            datapoint = self.data[index]
        else:
            datapoint = self.dict_data[index]
        item = {**datapoint}
        if self.return_frames:
            video = self._video_path_handler.video_from_path(
                self.frames_dir / datapoint["frame_path"]
            )
            # just get the whole video since the clip is already extracted
            clip = video.get_clip(0, video.duration)

            # pytorch video returns pixels as float by default, which causes
            # problems downstream, so let's convert them to uint8.
            item["video"] = clip["video"].to(torch.uint8)

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self.data)


class FrameInterleavedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frames_dir: str,
        annotation_file: str | None = None,
        in_context_example_frames_dir: str | None = None,
        in_context_example_annotation_file: str | None = None,
        num_in_context_examples_per_sample: int = 4,
        verb_noun_ratio: float = 0.5,
        transform: Callable[[dict], Any] | None = None,
        return_frames: bool = True,
        random_in_context_examples: bool = False,
        target_dataset_len: int | None = None,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param annotation_file: path to annotation file. If frames_dir contains
            narrated_actions.csv, this is optional.
        :param in_context_example_frames_dir: path to dir that contains
            extracted frames for in-context examples.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param in_context_example_annotation_file: path to annotation file for
            in-context examples. If in_context_example_frames_dir contains
            narrated_actions.csv, this is optional.
        :param num_in_context_examples_per_sample: number of in-context examples to
            sample pere datapoint
        :param verb_noun_ratio: target verb/noun ratio for in-context examples
        :param transform: transform function to be called for each datapoint
        :param return_frames: whether to return frame data for each datapoint or not
        :param random_in_context_examples: whether to sample random in-context examples
            or not
        :param target_dataset_len: if given, we upsample datapoints based on their
            verb/noun classes to match target_dataset_len
        """
        self.num_in_context_examples_per_sample = num_in_context_examples_per_sample
        self.verb_noun_ratio = verb_noun_ratio
        self.return_frames = return_frames
        self.random_in_context_examples = random_in_context_examples
        self.target_dataset_len = target_dataset_len
        self._dataset = FrameDataset(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            return_frames=return_frames,
        )
        if self.target_dataset_len is not None and self.target_dataset_len > len(
            self._dataset
        ):
            action_buckets: dict[tuple[str, str], set[int]] = defaultdict(set)
            for i, datapoint in enumerate(self._dataset.data):
                action_buckets[
                    (datapoint["structured_verb"], datapoint["structured_noun"])
                ].add(i)
            num_to_sample_per_action = (
                self.target_dataset_len - len(self._dataset)
            ) // len(action_buckets)
            for _, idx in action_buckets.items():
                if len(self._dataset) == self.target_dataset_len:
                    break
                num_to_sample = max(
                    num_to_sample_per_action,
                    len(self._dataset) - self.target_dataset_len,
                )
                sampled_idx: list[int] = []
                while len(sampled_idx) < num_to_sample:
                    curr_num_to_sample = num_to_sample - len(sampled_idx)
                    if len(idx) >= curr_num_to_sample:
                        sampled_idx.extend(random.sample(idx, curr_num_to_sample))
                    else:
                        sampled_idx.extend(idx)
                for i in sampled_idx:
                    sampled = self._dataset.data[i]
                    self._dataset.data.append(sampled)
                    self._dataset.dict_data[sampled["frame_path"]] = sampled
        if in_context_example_frames_dir is None:
            self.in_context_examples_from_main_dataset = True
            self._in_context_dataset = self._dataset
        else:
            self.in_context_examples_from_main_dataset = False
            self._in_context_dataset = FrameDataset(
                in_context_example_frames_dir,
                annotation_file=in_context_example_annotation_file,
                return_frames=return_frames,
            )

        self.structured_verb_buckets: dict[str, set[int]] = defaultdict(set)
        self.structured_noun_buckets: dict[str, set[int]] = defaultdict(set)
        if not self.random_in_context_examples:
            # put datapoints into buckets based on their structured verbs and nouns
            for i, datapoint in enumerate(self._in_context_dataset.data):
                if datapoint["structured_verb"] not in {"", "[other]"}:
                    # NOTE: [other] is a catch-all verb in Ego4D. For these instances,
                    # we just sample from the whole dataset.
                    self.structured_verb_buckets[datapoint["structured_verb"]].add(i)
                if datapoint["structured_noun"] != "":
                    self.structured_noun_buckets[datapoint["structured_noun"]].add(i)

        self._transform = transform

    def _sample_in_context_examples_based_on_structured_verb_noun(
        self, datapoint: dict[str, Any], index: int
    ) -> set[int]:
        verb_bucket: set[int] = set()
        for i in self.structured_verb_buckets.get(datapoint["structured_verb"], set()):
            if self.in_context_examples_from_main_dataset and i == index:
                # filter out the current example if the in-context example
                # dataset is the same as the main dataset
                continue
            if (
                self._in_context_dataset.data[i]["structured_noun"]
                == datapoint["structured_noun"]
            ):
                # if this in-context example candidate has the same verb and noun
                # as the current example, skip it.
                continue
            verb_bucket.add(i)
        noun_bucket: set[int] = set()
        for i in self.structured_noun_buckets.get(datapoint["structured_noun"], set()):
            if self.in_context_examples_from_main_dataset and i == index:
                # filter out the current example if the in-context example
                # dataset is the same as the main dataset
                continue
            if (
                self._in_context_dataset.data[i]["structured_verb"]
                == datapoint["structured_verb"]
            ):
                # if this in-context example candidate has the same verb and noun
                # as the current example, skip it.
                continue
            noun_bucket.add(i)

        def _sample(bucket: set[int], k: int) -> set[int]:
            if len(bucket) >= k:
                samples = set(random.sample(bucket, k))
            else:
                samples = set(bucket)
            bucket -= samples
            return samples

        examples: set[int] = set()
        num_additional_examples = self.num_in_context_examples_per_sample - len(
            examples
        )
        while num_additional_examples > 0 and (
            len(verb_bucket) > 0 or len(noun_bucket) > 0
        ):
            if len(verb_bucket) > 0 and len(noun_bucket) > 0:
                num_verb_examples = int(num_additional_examples * self.verb_noun_ratio)
                num_noun_examples = num_additional_examples - num_verb_examples
            elif len(verb_bucket) == 0:
                num_verb_examples = 0
                num_noun_examples = num_additional_examples
            else:
                num_noun_examples = 0
                num_verb_examples = num_additional_examples

            examples |= _sample(verb_bucket, num_verb_examples)
            examples |= _sample(noun_bucket, num_noun_examples)
            num_additional_examples = self.num_in_context_examples_per_sample - len(
                examples
            )

        if num_additional_examples > 0:
            # there wasn't enough samples in verb and noun buckets, so sample from the
            # rest of the dataset
            rest: set[int] = set()
            for i in range(len(self._in_context_dataset)):
                if (self.in_context_examples_from_main_dataset and i == index) or (
                    i in examples
                ):
                    # filter out the current example if the in-context example
                    # dataset is the same as the main dataset or
                    # it's already been drawn.
                    continue
                if (
                    self._in_context_dataset.data[i]["structured_verb"]
                    == datapoint["structured_verb"]
                    and self._in_context_dataset.data[i]["structured_noun"]
                    == datapoint["structured_noun"]
                ):
                    # if this in-context example candidate has the same verb and noun
                    # as the current example, skip it.
                    continue
                rest.add(i)
            examples |= _sample(rest, num_additional_examples)

        return examples

    def __getitem__(self, index: int) -> dict[str, Any]:
        datapoint = self._dataset[index]

        if self.random_in_context_examples:
            examples = set(
                random.sample(
                    [
                        i
                        for i in range(len(self._in_context_dataset))
                        if not self.in_context_examples_from_main_dataset or i != index
                    ],
                    self.num_in_context_examples_per_sample,
                )
            )
            item = {
                "items": [self._in_context_dataset[i] for i in examples] + [datapoint]
            }
        else:
            examples = self._sample_in_context_examples_based_on_structured_verb_noun(
                datapoint, index
            )

            item = {
                "items": [
                    self._in_context_dataset[i]
                    # shuffle the in-context examples and
                    # append the main datapoint in the end
                    for i in random.sample(examples, len(examples))
                ]
                + [datapoint]
            }

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._dataset)


class FrameInterleavedPresampledDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frames_dir: str,
        in_context_query_map_file_path: str,
        in_context_example_frames_dir: str,
        annotation_file: str | None = None,
        in_context_example_annotation_file: str | None = None,
        transform: Callable[[dict], Any] | None = None,
        return_frames: bool = True,
        shuffle_in_context_example_frames: bool = False,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param in_context_query_map_file_path: path to file that specifies
            the mapping between in-context examples and queries.
        :param in_context_example_frames_dir: path to dir that contains
            extracted frames for in-context examples.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param annotation_file: path to annotation file. If frames_dir contains
            narrated_actions.csv, this is optional.
        :param in_context_example_annotation_file: path to annotation file for
            in-context examples. If in_context_example_frames_dir contains
            narrated_actions.csv, this is optional.
        :param transform: transform function to be called for each datapoint
        :param return_frames: whether to return frame data for each datapoint or not
        :param shuffle_in_context_example_frames: shuffle video frames of in-context
            examples. This option actually generates "permutations with no fixed points"
            or "derangements" (https://en.wikipedia.org/wiki/Derangement).
            Useful for ablation studies.
        """
        self.return_frames = return_frames
        self.shuffle_in_context_example_frames = shuffle_in_context_example_frames
        self._transform = transform
        self._dataset = FrameDataset(
            frames_dir, annotation_file=annotation_file, return_frames=return_frames
        )
        self._in_context_dataset = FrameDataset(
            in_context_example_frames_dir,
            annotation_file=in_context_example_annotation_file,
            return_frames=return_frames,
        )
        self._in_context_query_map: list[dict[str, Any]] = []
        with open(in_context_query_map_file_path) as f:
            for line in f:
                self._in_context_query_map.append(json.loads(line))

    def __getitem__(self, index: int) -> dict[str, Any]:
        in_context_query = self._in_context_query_map[index]
        in_context_examples = [
            self._in_context_dataset[in_context_example]
            for in_context_example in in_context_query["context"]
        ]
        if self.shuffle_in_context_example_frames:
            video_idx = list(range(len(in_context_examples)))
            shuffled_video_idx = video_idx[:]
            while True:
                # we basically shuffle until no videos are in their original positions
                # The probability that a random permutation is a derangement is
                # approximately 1/e no matter how long the list is. As a result,
                # the expected number of necessary shuffles is about 3, and it rarely
                # goes over more then ten.
                # https://stackoverflow.com/questions/15512058/python-shuffle-such-that-position-will-never-repeat/
                random.shuffle(shuffled_video_idx)
                for a, b in zip(video_idx, shuffled_video_idx):
                    if a == b:
                        break
                else:
                    # this else clause is only executed when we exit the for loop
                    # "normally" without encountering a break statement.
                    # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
                    # we break out of the while loop here since we've found a
                    # derangement.
                    break
            shuffled_videos = [
                in_context_examples[idx]["video"] for idx in shuffled_video_idx
            ]
            for example, frames in zip(in_context_examples, shuffled_videos):
                example["video"] = frames
        item = {
            "items": in_context_examples + [self._dataset[in_context_query["query"]]]
        }
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._in_context_query_map)
