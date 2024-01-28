import argparse
from typing import Any

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    Permute,
    UniformTemporalSubsample,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Resize
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from eilev.data.frame import FrameDataset


class VerbNounClassifier(nn.Module):
    def __init__(
        self,
        verb_classifier: VideoMAEForVideoClassification,
        noun_classifier: VideoMAEForVideoClassification,
    ) -> None:
        super().__init__()
        self.verb_classifier = verb_classifier
        self.noun_classifier = noun_classifier

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        verb_output = self.verb_classifier(**kwargs, return_dict=True)
        noun_output = self.noun_classifier(**kwargs, return_dict=True)
        return {"verb_logits": verb_output.logits, "noun_logits": noun_output.logits}


@torch.no_grad()
def predict(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: VerbNounClassifier | DistributedDataParallel,
    log_to_wandb: bool,
    print_predictions: bool,
    num_predict_batches: int | None,
) -> None:
    if isinstance(model, DistributedDataParallel):
        module = model.module
    else:
        module = model
    dtype = module.verb_classifier.dtype
    device = module.verb_classifier.device
    if log_to_wandb:
        table = wandb.Table(
            columns=[
                "frame_path",
                "video_uid",
                "clip_index",
                "predicted_structured_verb",
                "ground_truth_structured_verb",
                "predicted_structured_noun",
                "ground_truth_structured_noun",
                "ground_truth_narration_text",
            ]
        )
    else:
        table = None
    for i, datapoint in enumerate(tqdm(dataloader, desc="Predicting")):
        if num_predict_batches is not None and i == num_predict_batches:
            break
        outputs = module(
            pixel_values=datapoint["pixel_values"].to(dtype=dtype, device=device)
        )
        verb_logits = accelerator.gather_for_metrics(outputs["verb_logits"])
        noun_logits = accelerator.gather_for_metrics(outputs["noun_logits"])
        frame_paths = gather_object(datapoint["frame_path"])
        video_uids = gather_object(datapoint["video_uid"])
        clip_indices = gather_object(datapoint["clip_index"])
        ground_truth_narration_texts = gather_object(datapoint["narration_text"])
        ground_truth_structured_verbs = gather_object(datapoint["structured_verb"])
        ground_truth_structured_nouns = gather_object(datapoint["structured_noun"])
        if (
            accelerator.gradient_state.end_of_dataloader
            and accelerator.gradient_state.remainder > 0
        ):
            # we have some duplicates, so filter them out
            # this logic is from gather_for_metrics()
            frame_paths = frame_paths[: accelerator.gradient_state.remainder]
            video_uids = video_uids[: accelerator.gradient_state.remainder]
            clip_indices = clip_indices[: accelerator.gradient_state.remainder]
            ground_truth_narration_texts = ground_truth_narration_texts[
                : accelerator.gradient_state.remainder
            ]
            ground_truth_structured_verbs = ground_truth_structured_verbs[
                : accelerator.gradient_state.remainder
            ]
            ground_truth_structured_nouns = ground_truth_structured_nouns[
                : accelerator.gradient_state.remainder
            ]
        predicted_verbs = [
            model.verb_classifier.config.id2label[idx.item()]  # type: ignore
            for idx in verb_logits.argmax(dim=1)
        ]
        predicted_nouns = [
            model.noun_classifier.config.id2label[idx.item()]  # type: ignore
            for idx in noun_logits.argmax(dim=1)
        ]
        if print_predictions:
            for (
                predicted_verb,
                ground_truth_structured_verb,
                predicted_noun,
                ground_truth_structured_noun,
                ground_truth_narration_text,
            ) in zip(
                predicted_verbs,
                ground_truth_structured_verbs,
                predicted_nouns,
                ground_truth_structured_nouns,
                ground_truth_narration_texts,
            ):
                print(
                    f"Predicted verb: {predicted_verb}, "
                    f"Ground-truth verb: {ground_truth_structured_verb}"
                )
                print(
                    f"Predicted noun: {predicted_noun}, "
                    f"Ground-truth noun: {ground_truth_structured_noun}"
                )
                print(f"Ground-truth narration text: {ground_truth_narration_text}")
        if table is not None:
            for (
                frame_path,
                video_uid,
                clip_index,
                predicted_verb,
                ground_truth_structured_verb,
                predicted_noun,
                ground_truth_structured_noun,
                ground_truth_narration_text,
            ) in zip(
                frame_paths,
                video_uids,
                clip_indices,
                predicted_verbs,
                ground_truth_structured_verbs,
                predicted_nouns,
                ground_truth_structured_nouns,
                ground_truth_narration_texts,
            ):
                table.add_data(
                    frame_path,
                    video_uid,
                    clip_index,
                    predicted_verb,
                    ground_truth_structured_verb,
                    predicted_noun,
                    ground_truth_structured_noun,
                    ground_truth_narration_text,
                )
    if table is not None:
        accelerator.log({"generated": table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verb_model", required=True)
    parser.add_argument("--noun_model", required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bfloat16"], default="fp32")
    parser.add_argument("--num_dataloader_workers", default=0, type=int)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--annotation_file")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_predict_batches", default=None, type=int)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--print_predictions", action="store_true")
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
    model = VerbNounClassifier(
        VideoMAEForVideoClassification.from_pretrained(
            args.verb_model, torch_dtype=dtype
        ),
        VideoMAEForVideoClassification.from_pretrained(
            args.noun_model, torch_dtype=dtype
        ),
    )

    processor = VideoMAEImageProcessor.from_pretrained(args.verb_model)
    if "shortest_edge" in processor.size:
        height = width = processor.size["shortest_edge"]
    else:
        height = processor.size["height"]
        width = processor.size["width"]
    dataset = FrameDataset(
        args.frames_dir,
        annotation_file=args.annotation_file,
        transform=ApplyTransformToKey(
            "video",
            transform=Compose(
                [
                    UniformTemporalSubsample(model.verb_classifier.config.num_frames),
                    # Can't use VideoMAEImageProcessor here b/c it doesn't
                    # play nicely with Tensors, e.g., creating a tensor from
                    # a list of numpy.ndarrays, which is extremely slow.
                    Lambda(lambda x: x * processor.rescale_factor),
                    Normalize(processor.image_mean, processor.image_std),
                    Resize((height, width), antialias=True),
                    Permute((1, 0, 2, 3)),
                ]
            ),
        ),
    )

    def collate_fn(examples: list[dict[str, Any]]):
        return {
            "pixel_values": torch.stack([example["video"] for example in examples]),
            "frame_path": [example["frame_path"] for example in examples],
            "video_uid": [example["video_uid"] for example in examples],
            "clip_index": [example["clip_index"] for example in examples],
            "narration_text": [example["narration_text"] for example in examples],
            "structured_verb": [example["structured_verb"] for example in examples],
            "structured_noun": [example["structured_noun"] for example in examples],
        }

    model, dataloader = accelerator.prepare(
        model,
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        ),
    )

    predict(
        accelerator,
        dataloader,
        model,
        args.wandb_project is not None,
        args.print_predictions,
        args.num_predict_batches,
    )
    if args.wandb_project is not None:
        accelerator.end_training()
