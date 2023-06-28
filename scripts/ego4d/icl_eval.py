import argparse
import json
from collections import defaultdict
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm
from transformers import Blip2Processor

from video_blip.data.ego4d import Ego4dFHOMainFrameDataset
from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    generate_input_ids_and_labels_from_interleaved,
)
from video_blip.model.utils import process
from video_blip.model.v2 import VideoBlipForConditionalGeneration


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
    dataset: Ego4dFHOMainFrameDataset,
    num_eval_datapoints: int,
) -> Ego4dFHOMainFrameDataset:
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
        self, processor: Blip2Processor, few_shot_prompt: str, eos_token_id: int
    ) -> None:
        self.processor = processor
        self.collator = DataCollatorForInterleavedVideoSeq2Seq(processor.tokenizer)
        self.few_shot_prompt = few_shot_prompt
        self.eos_token_id = eos_token_id

    def preprocess(
        self,
        classes: list[str],
        prompt: str,
        datapoint: dict[str, Any],
        few_shot_examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        few_shot_prompts: list[tuple[str, str | None]] = [
            (
                " ".join(
                    [
                        self.few_shot_prompt,
                        example["structured_verb"],
                        example["structured_noun"],
                    ]
                ),
                "",
            )
            for example in few_shot_examples
        ]
        inputs_list = [
            generate_input_ids_and_labels_from_interleaved(
                self.processor.tokenizer,
                self.eos_token_id,
                few_shot_prompts + [(prompt, cls)],
                len(few_shot_examples) + 1,
                [[i] for i in range(len(few_shot_examples) + 1)],
            )
            for cls in classes
        ]
        # input_ids: (num_classes, seq_len)
        # attention_mask: (num_classes, seq_len)
        # labels: (num_classes, seq_len)
        # video_causal_mask: (num_classes, seq_len, num_videos)
        inputs = self.collator(inputs_list)

        # (num_videos, channel, time, height, width)
        pixel_values = process(
            self.processor,
            video=torch.stack(
                [item["video"] for item in few_shot_examples + [datapoint]]
            ),
        ).pixel_values

        return {"pixel_values": pixel_values, **inputs}


def eval(
    eval_dataset: Ego4dFHOMainFrameDataset,
    train_dataset: Ego4dFHOMainFrameDataset,
    num_shot: int,
    num_dataloader_workers: int,
    num_few_shot_dataloader_workers: int,
    model: VideoBlipForConditionalGeneration,
    preprocessor: Preprocessor,
    structured_verbs: list[str],
    structured_nouns: list[str],
    log_verb_preds: bool,
    print_verb_preds: bool,
) -> None:
    verb_id_map = {verb: i for i, verb in enumerate(structured_verbs)}
    verb_f1 = MulticlassF1Score(len(structured_verbs))
    if log_verb_preds:
        verb_pred_table = wandb.Table(
            columns=[
                "frame_path",
                "video_uid",
                "clip_index",
                "structured_verb",
                "prediction",
            ]
        )
    else:
        verb_pred_table = None
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
            structured_verbs, preprocessor.few_shot_prompt, datapoint, few_shot_examples
        )
        preprocessed["pixel_values"] = preprocessed["pixel_values"].to(
            dtype=model.dtype, device=model.device
        )
        preprocessed["video_causal_mask"] = preprocessed["video_causal_mask"].to(
            device=model.device
        )
        preprocessed["input_ids"] = preprocessed["input_ids"].to(device=model.device)
        preprocessed["attention_mask"] = preprocessed["attention_mask"].to(
            device=model.device
        )
        preprocessed["labels"] = preprocessed["labels"].to(device=model.device)
        # (num_structured_verbs)
        log_likelihood = model.classify(**preprocessed)
        verb_f1(
            log_likelihood.unsqueeze(0).to("cpu"),
            torch.tensor([verb_id_map[datapoint["structured_verb"]]]),
        )
        pred_verb = structured_verbs[log_likelihood.argmax()]
        if print_verb_preds:
            print(
                f'Predicted: {pred_verb}, Ground Truth: {datapoint["structured_verb"]}'
            )
        if verb_pred_table is not None:
            verb_pred_table.add_data(
                datapoint["frame_path"],
                datapoint["video_uid"],
                datapoint["clip_index"],
                datapoint["structured_verb"],
                pred_verb,
            )
    wandb_log_dict: dict[str, Any] = {"verb_f1": verb_f1.compute()}
    print(f"Verb F1: {wandb_log_dict['verb_f1']}")
    if verb_pred_table is not None:
        wandb_log_dict["verb_pred_table"] = verb_pred_table
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
    parser.add_argument("--train_narrated_actions_dir", required=True)
    parser.add_argument("--eval_narrated_actions_dir", required=True)
    parser.add_argument("--num_shot", required=True, type=int)
    parser.add_argument("--log_verb_preds", action="store_true")
    parser.add_argument("--print_verb_preds", action="store_true")
    parser.add_argument("--num_eval_datapoints", default=0, type=int)
    parser.add_argument("--random-seed", type=int, default=42)
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
        Ego4dFHOMainFrameDataset(args.train_narrated_actions_dir),
        0,
    )
    eval_dataset = add_and_filter_verb_noun(
        narrated_action_verb_noun,
        Ego4dFHOMainFrameDataset(args.eval_narrated_actions_dir),
        args.num_eval_datapoints,
    )

    preprocessor = Preprocessor(
        processor,
        "Question: What is the camera wearer doing? Answer:",
        model.config.text_config.eos_token_id,
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
        fho_lta_taxonomy["nouns"],
        args.log_verb_preds,
        args.print_verb_preds,
    )
