from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
import transformers
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose
from transformers import Blip2Processor
from transformers.deepspeed import is_deepspeed_zero3_enabled

from video_blip.data.ego4d import Ego4dFHOMainFrameInterleavedDataset
from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels_from_interleaved,
)
from video_blip.model.v2 import VideoBlipForConditionalGeneration

PROMPT = "Question: What is the camera wearer doing? Answer:"


def preprocess(
    processor: Blip2Processor,
    datapoint: dict[str, Any],
    decoder_only_lm: bool = True,
    video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    cleaned_narration_texts: list[str] = []
    for item in datapoint["items"]:
        cleaned_narration_texts.append(clean_narration_text(item["narration_text"]))
    preprocessed = generate_input_ids_and_labels_from_interleaved(
        processor.tokenizer,
        [PROMPT for _ in range(len(datapoint["items"]))],
        cleaned_narration_texts,
        len(datapoint["items"]),
        [[i] for i in range(len(datapoint["items"]))],
        decoder_only_lm,
    )
    videos = [item["video"] for item in datapoint["items"]]
    if video_transform is not None:
        for i in range(len(videos)):
            videos[i] = video_transform(videos[i])
    preprocessed["pixel_values"] = torch.stack(videos)

    return preprocessed


# NOTE: We can't use 3.10's new X|Y syntax b/c HfArgumentParser doesn't support it.
# https://github.com/huggingface/transformers/issues/20249
@dataclass
class ModelArguments:
    model_name_or_path: str
    num_subsample_frames: int


@dataclass
class DataArguments:
    train_narrated_actions_dir: str
    val_narrated_actions_dir: str
    num_videos_per_sample: int


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False

    processor = transformers.Blip2Processor.from_pretrained(
        model_args.model_name_or_path
    )
    model = VideoBlipForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
    )
    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()

    train_data = Ego4dFHOMainFrameInterleavedDataset(
        data_args.train_narrated_actions_dir,
        num_videos_per_sample=data_args.num_videos_per_sample,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model,
            video_transform=Compose(
                [UniformTemporalSubsample(model_args.num_subsample_frames)]
            ),
        ),
    )
    val_data = Ego4dFHOMainFrameInterleavedDataset(
        data_args.val_narrated_actions_dir,
        num_videos_per_sample=1,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model,
            video_transform=Compose(
                [UniformTemporalSubsample(model_args.num_subsample_frames)]
            ),
        ),
    )

    # Load the best model at the end so we can save it
    training_args.load_best_model_at_end = True

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForInterleavedVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
