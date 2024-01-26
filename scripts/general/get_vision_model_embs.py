import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import Blip2Processor

from eilev.data.frame import FrameDataset
from eilev.model.utils import process
from eilev.model.v2 import VideoBlipForConditionalGeneration, VideoBlipVisionModel


@dataclass
class Preprocessor:
    processor: Blip2Processor

    def __call__(self, datapoint: dict[str, Any]) -> dict[str, Any]:
        pixel_values = process(
            self.processor, video=datapoint["video"]
        ).pixel_values.squeeze(0)
        return {
            # (channel, time, height, width)
            "pixel_values": pixel_values,
            **datapoint,
        }


def eval(
    accelerator: Accelerator,
    dataloader: DataLoader,
    output_dir: str,
    model: VideoBlipVisionModel | DistributedDataParallel,
    num_eval_datapoints: int | None,
) -> None:
    if isinstance(model, DistributedDataParallel):
        dtype = model.module.dtype
        module = model.module
    else:
        dtype = model.dtype
        module = model
    device = model.device

    for i, datapoint in enumerate(tqdm(dataloader, desc="Calculating")):
        if num_eval_datapoints is not None and i == num_eval_datapoints:
            break

        # (batch, hidden_size)
        with torch.no_grad():
            mean_pooler_output = module(
                pixel_values=datapoint["pixel_values"].to(dtype=dtype, device=device),
                return_dict=True,
            ).pooler_output.mean(dim=1)
        all_mean_pooler_output = accelerator.gather_for_metrics(mean_pooler_output)
        all_frame_paths = gather_object(datapoint["frame_paths"])
        for emb, frame_path in zip(all_mean_pooler_output.to("cpu"), all_frame_paths):
            # Note that frame_paths may have duplicates in the end, but zip()
            # automatically drops them according to the length of
            # all_mean_pooler_output.
            torch.save(emb, Path(output_dir) / (frame_path + ".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bfloat16"], default="fp32")
    parser.add_argument("--num_dataloader_workers", default=0, type=int)
    parser.add_argument("--frames_dirs", required=True, nargs="+")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_eval_datapoints", default=None, type=int)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

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
    ).vision_model
    if args.processor is None:
        args.processor = args.model

    processor = Blip2Processor.from_pretrained(args.processor)
    datasets = [
        FrameDataset(frames_dir, transform=Preprocessor(processor))
        for frames_dir in args.frames_dirs
    ]

    def collate_fn(examples: list[dict[str, Any]]):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        frame_paths = [example["frame_path"] for example in examples]
        video_uids = [example["video_uid"] for example in examples]
        clip_indices = [example["clip_index"] for example in examples]
        return {
            "pixel_values": pixel_values,
            "frame_paths": frame_paths,
            "video_uids": video_uids,
            "clip_indices": clip_indices,
        }

    model, dataloader = accelerator.prepare(
        model,
        DataLoader(
            ConcatDataset(datasets),
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        ),
    )

    eval(
        accelerator,
        dataloader,
        args.output_dir,
        model,
        args.num_eval_datapoints,
    )
