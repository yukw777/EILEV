import argparse
import json
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import Blip2Processor

from video_blip.data.ego4d import Ego4dFHOMainFrameDataset
from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels_from_interleaved,
)
from video_blip.model.utils import process
from video_blip.model.v2 import VideoBlipForConditionalGeneration


class Preprocessor:
    def __init__(self, processor: Blip2Processor, prompt: str) -> None:
        self.processor = processor
        self.collator = DataCollatorForInterleavedVideoSeq2Seq(processor.tokenizer)
        self.prompt = prompt

    def preprocess(
        self, datapoint: dict[str, Any], few_shot_examples: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        few_shot_prompts: list[tuple[str, str | None]] = [
            (self.prompt + " " + clean_narration_text(example["narration_text"]), "")
            for example in few_shot_examples
        ]
        # input_ids: (prompt_seq_len)
        # labels: (prompt_seq_len)
        # video_causal_mask: (prompt_seq_len, num_videos)
        inputs = generate_input_ids_and_labels_from_interleaved(
            self.processor.tokenizer,
            few_shot_prompts + [(self.prompt, None)],
            len(few_shot_examples) + 1,
            [[i] for i in range(len(few_shot_examples) + 1)],
        )
        # (num_videos, channel, time, height, width)
        pixel_values = process(
            self.processor,
            video=torch.stack(
                [item["video"] for item in few_shot_examples + [datapoint]]
            ),
        ).pixel_values

        return {
            # (num_videos, channel, time, height, width)
            "pixel_values": pixel_values,
            **inputs,
        }


def eval(
    eval_dataset: Ego4dFHOMainFrameDataset,
    train_dataset: Ego4dFHOMainFrameDataset,
    num_shot: int,
    num_dataloader_workers: int,
    num_few_shot_dataloader_workers: int,
    model: VideoBlipForConditionalGeneration,
    preprocessor: Preprocessor,
    print_narration_texts: bool,
    log_narration_texts: bool,
    generation_config: dict,
) -> None:
    if log_narration_texts:
        table = wandb.Table(
            columns=[
                "frame_path",
                "video_uid",
                "clip_index",
                "generated",
                "ground_truth",
            ]
        )
    else:
        table = None
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
        preprocessed = preprocessor.preprocess(datapoint, few_shot_examples)
        generated_ids = model.generate(
            pixel_values=preprocessed["pixel_values"]
            .to(dtype=model.dtype, device=model.device)
            .unsqueeze(0),
            input_ids=preprocessed["input_ids"].to(device=model.device).unsqueeze(0),
            video_causal_mask=preprocessed["video_causal_mask"]
            .to(device=model.device)
            .unsqueeze(0),
            **generation_config,
        )
        generated_text = preprocessor.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        ground_truth = clean_narration_text(datapoint["narration_text"])
        if print_narration_texts:
            print(f"Generated text: {generated_text}")
            print(f"Ground-truth text: {ground_truth}")
        if table is not None:
            table.add_data(
                datapoint["frame_path"],
                datapoint["video_uid"],
                datapoint["clip_index"],
                generated_text,
                ground_truth,
            )
    if table is not None:
        wandb.log({"generated": table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bfloat16"], default="fp32")
    parser.add_argument("--num_dataloader_workers", default=0, type=int)
    parser.add_argument("--num_few_shot_dataloader_workers", default=0, type=int)
    parser.add_argument("--train_narrated_actions_dir", required=True)
    parser.add_argument("--eval_narrated_actions_dir", required=True)
    parser.add_argument("--num_shot", required=True, type=int)
    parser.add_argument("--print_narration_texts", action="store_true")
    parser.add_argument("--log_narration_texts", action="store_true")
    parser.add_argument("--num_eval_datapoints", default=0, type=int)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--generation_config", default="{}")
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

    train_dataset = Ego4dFHOMainFrameDataset(args.train_narrated_actions_dir)
    eval_dataset = Ego4dFHOMainFrameDataset(args.eval_narrated_actions_dir)
    if args.num_eval_datapoints > 0 and len(eval_dataset) > args.num_eval_datapoints:
        eval_dataset.data = eval_dataset.data[: args.num_eval_datapoints]

    preprocessor = Preprocessor(
        processor, "Question: What is the camera wearer doing? Answer:"
    )
    eval(
        eval_dataset,
        train_dataset,
        args.num_shot,
        args.num_dataloader_workers,
        args.num_few_shot_dataloader_workers,
        model,
        preprocessor,
        args.print_narration_texts,
        args.log_narration_texts,
        json.loads(args.generation_config),
    )
