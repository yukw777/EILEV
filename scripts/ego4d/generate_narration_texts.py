import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Blip2Processor

from video_blip.data.ego4d import Ego4dFHOMainFrameInterleavedDataset
from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels_from_interleaved,
)
from video_blip.model.utils import process
from video_blip.model.v2 import VideoBlipForConditionalGeneration


class DataCollator(DataCollatorForInterleavedVideoSeq2Seq):
    def __call__(self, features, return_tensors=None):
        narration_texts = [feature.pop("narration_text") for feature in features]
        frame_paths = [feature.pop("frame_path") for feature in features]
        video_uids = [feature.pop("video_uid") for feature in features]
        clip_indexes = [feature.pop("clip_index") for feature in features]

        collated = super().__call__(features, return_tensors)

        collated["narration_text"] = narration_texts
        collated["frame_path"] = frame_paths
        collated["video_uid"] = video_uids
        collated["clip_index"] = clip_indexes
        return collated


@dataclass
class Preprocessor:
    processor: Blip2Processor
    prompt: str

    def __call__(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        inputs = generate_input_ids_and_labels_from_interleaved(
            self.processor.tokenizer,
            [
                (self.prompt + " " + clean_narration_text(item["narration_text"]), "")
                for item in datapoint["items"][:-1]
            ]
            + [(self.prompt, None)],
            len(datapoint["items"]),
            [[i] for i in range(len(datapoint["items"]))],
        )
        pixel_values = process(
            self.processor,
            video=torch.stack([item["video"] for item in datapoint["items"]]),
        ).pixel_values
        eval_item = datapoint["items"][-1]
        return {
            "narration_text": clean_narration_text(eval_item["narration_text"]),
            "frame_path": eval_item["frame_path"],
            "video_uid": eval_item["video_uid"],
            "clip_index": eval_item["clip_index"],
            # (num_videos, channel, time, height, width)
            "pixel_values": pixel_values,
            **inputs,
        }


def eval(
    accelerator: Accelerator,
    eval_dataloader: DataLoader,
    model: VideoBlipForConditionalGeneration | DistributedDataParallel,
    processor: Blip2Processor,
    use_video_causal_mask: bool,
    print_narration_texts: bool,
    log_narration_texts: bool,
    generation_config: dict,
    num_eval_datapoints: int | None,
) -> None:
    if isinstance(model, DistributedDataParallel):
        pass
        dtype = model.module.dtype
        module = model.module
    else:
        dtype = model.dtype
    device = model.device
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
    for i, datapoint in enumerate(tqdm(eval_dataloader, desc="Generating")):
        if num_eval_datapoints is not None and i == num_eval_datapoints:
            break
        generate_kwargs = {
            "pixel_values": datapoint["pixel_values"].to(dtype=dtype, device=device),
            "input_ids": datapoint["input_ids"].to(device=device),
            "attention_mask": datapoint["attention_mask"].to(device=device),
            **generation_config,
        }
        if use_video_causal_mask:
            generate_kwargs["video_causal_mask"] = datapoint["video_causal_mask"].to(
                device=device
            )
        generated_ids = module.generate(**generate_kwargs)
        generated_ids = accelerator.pad_across_processes(
            generated_ids, dim=1, pad_index=processor.tokenizer.pad_token_id
        )
        all_generated_ids = accelerator.gather_for_metrics(generated_ids)
        frame_paths = gather_object(datapoint["frame_path"])
        video_uids = gather_object(datapoint["video_uid"])
        clip_indices = gather_object(datapoint["clip_index"])
        ground_truth_texts = gather_object(datapoint["narration_text"])
        generated_texts = [
            text.strip()
            for text in processor.batch_decode(
                all_generated_ids, skip_special_tokens=True
            )
        ]
        if print_narration_texts:
            for generated_text, ground_truth_text in zip(
                generated_texts, ground_truth_texts
            ):
                print(f"Generated text: {generated_text}")
                print(f"Ground-truth text: {ground_truth_text}")
        if table is not None:
            for (
                frame_path,
                video_uid,
                clip_index,
                generated_text,
                ground_truth_text,
            ) in zip(
                frame_paths,
                video_uids,
                clip_indices,
                generated_texts,
                ground_truth_texts,
            ):
                table.add_data(
                    frame_path,
                    video_uid,
                    clip_index,
                    generated_text,
                    ground_truth_text,
                )
    if table is not None:
        accelerator.log({"generated": table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bfloat16"], default="fp32")
    parser.add_argument("--num_dataloader_workers", default=0, type=int)
    parser.add_argument("--train_narrated_actions_dir", required=True)
    parser.add_argument("--eval_narrated_actions_dir", required=True)
    parser.add_argument("--num_shot", required=True, type=int)
    parser.add_argument("--verb_noun_ratio", required=True, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--no_video_causal_mask", action="store_true")
    parser.add_argument("--print_narration_texts", action="store_true")
    parser.add_argument("--num_eval_datapoints", default=None, type=int)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--generation_config", default='{"max_new_tokens": 512}')
    parser.add_argument("--wandb_project")
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    if args.wandb_project is not None:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(args.wandb_project, config=args)
    else:
        accelerator = Accelerator()

    # initialize model and processor
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_dict[args.dtype]
    model = VideoBlipForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if args.processor is None:
        args.processor = args.model

    # in order to support batch generation, we need to pad on the left side
    processor = Blip2Processor.from_pretrained(args.processor, padding_side="left")
    eval_dataset = Ego4dFHOMainFrameInterleavedDataset(
        args.eval_narrated_actions_dir,
        in_context_example_narrated_actions_dir=args.train_narrated_actions_dir,
        num_in_context_examples_per_sample=args.num_shot,
        verb_noun_ratio=args.verb_noun_ratio,
        transform=Preprocessor(
            processor, "Question: What is the camera wearer doing? Answer:"
        ),
    )
    model, eval_dataloader = accelerator.prepare(
        model,
        DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            collate_fn=DataCollator(processor.tokenizer),
            pin_memory=True,
        ),
    )

    generation_config = json.loads(args.generation_config)
    if "max_new_tokens" not in generation_config:
        generation_config["max_new_tokens"] = 512
    eval(
        accelerator,
        eval_dataloader,
        model,
        processor,
        not args.no_video_causal_mask,
        args.print_narration_texts,
        args.wandb_project is not None,
        generation_config,
        args.num_eval_datapoints,
    )
    accelerator.end_training()
