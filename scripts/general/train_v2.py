import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from pytorchvideo.transforms import (
    ConvertUint8ToFloat,
    Normalize,
    Permute,
    RandAugment,
    RandomResizedCrop,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import PreTrainedTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from eilev.data.frame import FrameInterleavedDataset
from eilev.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels_from_interleaved,
)
from eilev.model.v2 import VideoBlipForConditionalGeneration

# Based on prompts from InstructBLIP
PROMPTS = [
    "What is the camera wearer doing?",
    "Question: What is the camera wearer doing?",
    "What is the camera wearer doing? An answer to the question is",
    "Q: What is the camera wearer doing? A:",
    "Given the video, answer the following question. What is the camera wearer doing?",
    "Based on the video, respond to this question: What is the camera wearer doing? "
    "Answer:",
    "Use the provided video to answer the question: What is the camera wearer doing?",
    'What is the answer to the following question? "What is the camera wearer doing?"',
    'The question "What is the camera wearer doing?" can be answered using the video. '
    "The answer is",
]


@dataclass
class Preprocessor:
    tokenizer: PreTrainedTokenizer
    num_query_tokens: int
    decoder_only_lm: bool
    video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, datapoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        preprocessed = generate_input_ids_and_labels_from_interleaved(
            self.tokenizer,
            [
                (
                    random.choice(PROMPTS)
                    + " "
                    + clean_narration_text(item["narration_text"]),
                    1,
                )
                for item in datapoint["items"][:-1]
            ]
            + [(random.choice(PROMPTS), 1)],
            clean_narration_text(datapoint["items"][-1]["narration_text"]),
            self.num_query_tokens,
            self.decoder_only_lm,
        )
        videos = [item["video"] for item in datapoint["items"]]
        if self.video_transform is not None:
            for i in range(len(videos)):
                videos[i] = self.video_transform(videos[i])
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
    train_frames_dir: str
    val_frames_dir: str
    train_num_in_context_examples_per_sample: int
    val_num_in_context_examples_per_sample: int
    verb_noun_ratio: float
    train_annotation_file: str = None  # type: ignore
    val_annotation_file: str = None  # type: ignore
    random_in_context_examples: bool = False
    train_target_dataset_len: int = None  # type: ignore


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
    # OPT-based BLILP2 changes its eos token from `</s>` to `\n` for generation.
    # Let's reset it back to the original OPT eos token from the tokenizer.
    model.config.text_config.eos_token_id = processor.tokenizer.eos_token_id
    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()

    train_data = FrameInterleavedDataset(
        data_args.train_frames_dir,
        annotation_file=data_args.train_annotation_file,
        num_in_context_examples_per_sample=data_args.train_num_in_context_examples_per_sample,  # noqa: E501
        verb_noun_ratio=data_args.verb_noun_ratio,
        random_in_context_examples=data_args.random_in_context_examples,
        target_dataset_len=data_args.train_target_dataset_len,
        transform=Preprocessor(
            processor.tokenizer,
            model.config.num_query_tokens,
            model.config.use_decoder_only_language_model,
            video_transform=Compose(
                [
                    UniformTemporalSubsample(model_args.num_subsample_frames),
                    # close to BlipImageTrainProcessor from LAVIS
                    Permute((1, 0, 2, 3)),
                    # pytorch_video's RandAugment doesn't allow you to pick
                    # augmentations, so it performs more augmentations than used by
                    # BlipImageTrainProcessor.
                    RandAugment(magnitude=5),
                    Permute((1, 0, 2, 3)),
                    ConvertUint8ToFloat(),
                    Normalize(
                        processor.image_processor.image_mean,
                        processor.image_processor.image_std,
                    ),
                    RandomResizedCrop(
                        processor.image_processor.size["height"],
                        processor.image_processor.size["width"],
                        (0.5, 1.0),
                        (3.0 / 4.0, 4.0 / 3.0),
                        interpolation="bicubic",
                    ),
                    RandomHorizontalFlip(),
                ]
            ),
        ),
    )
    val_data = FrameInterleavedDataset(
        data_args.val_frames_dir,
        annotation_file=data_args.val_annotation_file,
        in_context_example_frames_dir=data_args.train_frames_dir,
        in_context_example_annotation_file=data_args.train_annotation_file,
        num_in_context_examples_per_sample=data_args.val_num_in_context_examples_per_sample,  # noqa: E501
        verb_noun_ratio=data_args.verb_noun_ratio,
        random_in_context_examples=data_args.random_in_context_examples,
        transform=Preprocessor(
            processor.tokenizer,
            model.config.num_query_tokens,
            model.config.use_decoder_only_language_model,
            video_transform=Compose(
                [
                    UniformTemporalSubsample(model_args.num_subsample_frames),
                    # Same as BlipImageProcessor from Hugging Face
                    Resize(
                        (
                            processor.image_processor.size["height"],
                            processor.image_processor.size["width"],
                        ),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    ConvertUint8ToFloat(),
                    Normalize(
                        processor.image_processor.image_mean,
                        processor.image_processor.image_std,
                    ),
                ]
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
