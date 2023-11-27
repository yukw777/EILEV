import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm
from transformers import Blip2Processor

from eilev.data.frame import FrameDataset
from eilev.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels_from_interleaved,
)
from eilev.model.utils import process
from eilev.model.v2 import VideoBlipForConditionalGeneration


def load_narrated_action_verb_noun(fho_main_path: str) -> dict[str, dict[str, str]]:
    with open(fho_main_path) as f:
        fho_main = json.load(f)

    narrated_action_verb_noun: dict[str, dict[str, str]] = defaultdict(dict)
    for video in fho_main["videos"]:
        for interval in video["annotated_intervals"]:
            for i, action in enumerate(interval["narrated_actions"]):
                if action["structured_verb"] in {"None", "[other]", "cross"}:
                    continue
                if action["frames"] is None:
                    continue
                for frame in action["frames"]:
                    if frame["frame_type"] != "pnr_frame":
                        # some actions don't have contact frames so use pnr_frame
                        continue
                    for box in frame["boxes"]:
                        if (
                            box["object_type"] == "object_of_change"
                            and box["structured_noun"] is not None
                        ):
                            narrated_action_verb_noun[
                                video["video_uid"] + "|" + str(i)
                            ] = {
                                "structured_verb": action["structured_verb"],
                                "structured_noun": box["structured_noun"],
                            }
                            break
    return narrated_action_verb_noun


def add_and_filter_verb_noun(
    narrated_action_verb_noun: dict[str, dict[str, str]],
    dataset: FrameDataset,
    num_eval_datapoints: int,
) -> FrameDataset:
    # if not in narrated_action_verb_noun, it's been filtered
    filtered_data = [
        datapoint
        for datapoint in dataset.data
        if datapoint["frame_path"] in narrated_action_verb_noun
    ]
    if num_eval_datapoints > 0 and len(filtered_data) > num_eval_datapoints:
        filtered_data = filtered_data[:num_eval_datapoints]

    # set structured verb and noun for each datapoint
    for datapoint in filtered_data:
        verb_noun = narrated_action_verb_noun[datapoint["frame_path"]]
        datapoint["structured_verb"] = verb_noun["structured_verb"]
        datapoint["structured_noun"] = verb_noun["structured_noun"]
    dataset.data = filtered_data
    return dataset


class Preprocessor:
    def __init__(
        self,
        processor: Blip2Processor,
        few_shot_prompt: str,
        num_query_tokens: int,
        use_decoder_only_lm: bool,
    ) -> None:
        self.processor = processor
        self.collator = DataCollatorForInterleavedVideoSeq2Seq(processor.tokenizer)
        self.few_shot_prompt = few_shot_prompt
        self.num_query_tokens = num_query_tokens
        self.use_decoder_only_lm = use_decoder_only_lm

    def preprocess(
        self,
        classes: list[str],
        prompt: str,
        datapoint: dict[str, Any],
        few_shot_examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        few_shot_prompts: list[tuple[str, int]] = [
            (
                " ".join(
                    [
                        self.few_shot_prompt,
                        clean_narration_text(example["narration_text"]),
                    ]
                ),
                1,
            )
            for example in few_shot_examples
        ]
        # input_ids: (prompt_seq_len)
        # labels: (prompt_seq_len)
        # video_input_mask: (prompt_seq_len)
        prompt_inputs = generate_input_ids_and_labels_from_interleaved(
            self.processor.tokenizer,
            few_shot_prompts + [(prompt, 1)],
            None,
            self.num_query_tokens,
            self.use_decoder_only_lm,
        )
        # input_ids: (num_classes, class_seq_len)
        # attention_mask: (num_classes, class_seq_len)
        class_inputs = self.processor.tokenizer(
            # prepend a space before each class name as the tokenizer
            # treats a word preceded by a space as a separate token.
            [" " + c for c in classes],
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )
        # (num_videos, channel, time, height, width)
        pixel_values = process(
            self.processor,
            video=torch.stack(
                [item["video"] for item in few_shot_examples + [datapoint]]
            ),
        ).pixel_values

        return {
            # (1, num_videos, channel, time, height, width)
            "pixel_values": pixel_values.unsqueeze(0),
            # (1, prompt_seq_len)
            "prompt_input_ids": prompt_inputs["input_ids"].unsqueeze(0),
            # (1, prompt_seq_len, num_videos)
            "prompt_video_input_mask": prompt_inputs["video_input_mask"].unsqueeze(0),
            # (num_classes, class_seq_len)
            "class_input_ids": class_inputs["input_ids"],
            # (num_classes, class_seq_len)
            "class_attention_mask": class_inputs["attention_mask"],
        }


def eval(
    eval_dataset: FrameDataset,
    train_dataset: FrameDataset,
    num_shot: int,
    num_dataloader_workers: int,
    num_few_shot_dataloader_workers: int,
    model: VideoBlipForConditionalGeneration,
    preprocessor: Preprocessor,
    structured_verbs: list[str],
    structured_verb_prompts: dict[str, str],
    structured_nouns: list[str],
    structured_noun_prompts: dict[str, str],
    log_verb_preds: bool,
    print_verb_preds: bool,
    log_noun_preds: bool,
    print_noun_preds: bool,
    class_batch_size: int,
) -> None:
    verb_prompts = list(structured_verb_prompts.keys())
    verb_id_map = {verb: i for i, verb in enumerate(structured_verbs)}
    verb_f1 = MulticlassF1Score(len(structured_verbs))
    if log_verb_preds:
        verb_pred_table = wandb.Table(
            columns=[
                "frame_path",
                "video_uid",
                "clip_index",
                "structured_verb",
                "predicted_verb_prompt",
                "prediction",
            ]
        )
    else:
        verb_pred_table = None

    if log_noun_preds:
        noun_pred_table = wandb.Table(
            columns=[
                "frame_path",
                "video_uid",
                "clip_index",
                "structured_noun",
                "predicted_noun_prompt",
                "prediction",
            ]
        )
    else:
        noun_pred_table = None

    noun_prompts = list(structured_noun_prompts.keys())
    noun_id_map = {noun: i for i, noun in enumerate(structured_nouns)}
    noun_f1 = MulticlassF1Score(len(structured_nouns))
    few_shot_dataloader_iter = (
        iter(
            DataLoader(
                train_dataset,
                sampler=RandomSampler(
                    # set num_samples to a high number so we can keep drawing few shot
                    # examples. Based on https://discuss.pytorch.org/t/infinite-random-sampler/30171/4 # noqa: E501
                    train_dataset,
                    replacement=True,
                    num_samples=int(1e10),
                ),
                batch_size=None,
                num_workers=num_few_shot_dataloader_workers,
                pin_memory=True,
            )
        )
        if num_shot > 0
        else None
    )
    for datapoint in tqdm(
        DataLoader(
            eval_dataset,
            batch_size=None,
            num_workers=num_dataloader_workers,
            pin_memory=True,
        ),
        desc="Evaluating",
    ):
        few_shot_examples = (
            [next(few_shot_dataloader_iter) for _ in range(num_shot)]
            if few_shot_dataloader_iter is not None
            else []
        )
        # First, classify verbs
        preprocessed = preprocessor.preprocess(
            verb_prompts,
            preprocessor.few_shot_prompt + " The camera wearer",
            datapoint,
            few_shot_examples,
        )
        preprocessed["pixel_values"] = preprocessed["pixel_values"].to(
            dtype=model.dtype, device=model.device
        )
        preprocessed["prompt_input_ids"] = preprocessed["prompt_input_ids"].to(
            device=model.device
        )
        preprocessed["prompt_video_input_mask"] = preprocessed[
            "prompt_video_input_mask"
        ].to(device=model.device)
        preprocessed["class_input_ids"] = preprocessed["class_input_ids"].to(
            device=model.device
        )
        preprocessed["class_attention_mask"] = preprocessed["class_attention_mask"].to(
            device=model.device
        )
        # (1, num_verb_prompts)
        log_likelihood = model.classify(
            **preprocessed, class_batch_size=class_batch_size
        )
        pred_verb_prompt = verb_prompts[log_likelihood.argmax(dim=-1)]
        pred_structured_verb = structured_verb_prompts[pred_verb_prompt]
        verb_f1(
            torch.tensor([verb_id_map[pred_structured_verb]]),
            torch.tensor([verb_id_map[datapoint["structured_verb"]]]),
        )
        if print_verb_preds:
            print(
                f"Predicted Verb: {pred_structured_verb}, "
                f'Ground Truth: {datapoint["structured_verb"]}'
            )
        if verb_pred_table is not None:
            verb_pred_table.add_data(
                datapoint["frame_path"],
                datapoint["video_uid"],
                datapoint["clip_index"],
                datapoint["structured_verb"],
                pred_verb_prompt,
                pred_structured_verb,
            )

        # Second, classify nouns
        preprocessed = preprocessor.preprocess(
            noun_prompts,
            preprocessor.few_shot_prompt + f" The camera wearer {pred_verb_prompt}",
            datapoint,
            few_shot_examples,
        )
        preprocessed["pixel_values"] = preprocessed["pixel_values"].to(
            dtype=model.dtype, device=model.device
        )
        preprocessed["prompt_input_ids"] = preprocessed["prompt_input_ids"].to(
            device=model.device
        )
        preprocessed["prompt_video_input_mask"] = preprocessed[
            "prompt_video_input_mask"
        ].to(device=model.device)
        preprocessed["class_input_ids"] = preprocessed["class_input_ids"].to(
            device=model.device
        )
        preprocessed["class_attention_mask"] = preprocessed["class_attention_mask"].to(
            device=model.device
        )
        # (1, num_verb_prompts)
        log_likelihood = model.classify(
            **preprocessed, class_batch_size=class_batch_size
        )
        pred_noun_prompt = noun_prompts[log_likelihood.argmax(dim=-1)]
        pred_structured_noun = structured_noun_prompts[pred_noun_prompt]
        noun_f1(
            torch.tensor([noun_id_map[pred_structured_noun]]),
            torch.tensor([noun_id_map[datapoint["structured_noun"]]]),
        )
        if print_noun_preds:
            print(
                f"Predicted Noun: {pred_structured_noun}, "
                f'Ground Truth: {datapoint["structured_noun"]}'
            )
        if noun_pred_table is not None:
            noun_pred_table.add_data(
                datapoint["frame_path"],
                datapoint["video_uid"],
                datapoint["clip_index"],
                datapoint["structured_noun"],
                pred_noun_prompt,
                pred_structured_noun,
            )
    wandb_log_dict: dict[str, Any] = {
        "verb_f1": verb_f1.compute(),
        "noun_f1": noun_f1.compute(),
    }
    print(f"Verb F1: {wandb_log_dict['verb_f1']}")
    print(f"Noun F1: {wandb_log_dict['noun_f1']}")
    if verb_pred_table is not None:
        wandb_log_dict["verb_pred_table"] = verb_pred_table
    if noun_pred_table is not None:
        wandb_log_dict["noun_pred_table"] = noun_pred_table
    wandb.log(wandb_log_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bfloat16"], default="fp32")
    parser.add_argument("--num_dataloader_workers", default=0, type=int)
    parser.add_argument("--num_few_shot_dataloader_workers", default=0, type=int)
    parser.add_argument("--fho_lta_taxonomy", required=True)
    parser.add_argument("--fho_main", required=True)
    parser.add_argument(
        "--structured_verb_prompt", default="eval-data/structured_verb_prompt.csv"
    )
    parser.add_argument(
        "--structured_noun_prompt", default="eval-data/structured_noun_prompt.csv"
    )
    parser.add_argument("--train_narrated_actions_dir", required=True)
    parser.add_argument("--eval_narrated_actions_dir", required=True)
    parser.add_argument("--num_shot", required=True, type=int)
    parser.add_argument("--log_verb_preds", action="store_true")
    parser.add_argument("--print_verb_preds", action="store_true")
    parser.add_argument("--log_noun_preds", action="store_true")
    parser.add_argument("--print_noun_preds", action="store_true")
    parser.add_argument("--num_eval_datapoints", default=0, type=int)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--class_batch_size", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    wandb.init(config=args)  # type: ignore

    # initialize model and processor
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_dict[args.dtype]
    model = VideoBlipForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    with open(args.fho_lta_taxonomy) as f:
        fho_lta_taxonomy = json.load(f)

    narrated_action_verb_noun = load_narrated_action_verb_noun(args.fho_main)
    train_dataset = add_and_filter_verb_noun(
        narrated_action_verb_noun,
        FrameDataset(args.train_narrated_actions_dir),
        0,
    )
    eval_dataset = add_and_filter_verb_noun(
        narrated_action_verb_noun,
        FrameDataset(args.eval_narrated_actions_dir),
        args.num_eval_datapoints,
    )

    with open(Path(__file__).parent / args.structured_verb_prompt, newline="") as f:
        reader = csv.DictReader(f)
        structured_verb_prompts: dict[str, str] = {}
        for row in reader:
            structured_verb_prompts[row["prompt"]] = row["structured_verb"]

    assert set(fho_lta_taxonomy["verbs"]) == set(structured_verb_prompts.values())

    with open(Path(__file__).parent / args.structured_noun_prompt, newline="") as f:
        reader = csv.DictReader(f)
        structured_noun_prompts: dict[str, str] = {}
        for row in reader:
            structured_noun_prompts[row["prompt"]] = row["structured_noun"]

    assert set(fho_lta_taxonomy["nouns"]) == set(structured_noun_prompts.values())

    preprocessor = Preprocessor(
        processor,
        "Question: What is the camera wearer doing? Answer:",
        model.config.num_query_tokens,
        model.config.use_decoder_only_language_model,
    )
    eval(
        eval_dataset,
        train_dataset,
        args.num_shot,
        args.num_dataloader_workers,
        args.num_few_shot_dataloader_workers,
        model,
        preprocessor,
        fho_lta_taxonomy["verbs"],
        structured_verb_prompts,
        fho_lta_taxonomy["nouns"],
        structured_noun_prompts,
        args.log_verb_preds,
        args.print_verb_preds,
        args.log_noun_preds,
        args.print_noun_preds,
        args.class_batch_size,
    )
