from dataclasses import dataclass
from typing import Any

import torch
import transformers
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    Permute,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from torchmetrics.functional.classification import multiclass_f1_score
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from eilev.data.frame import FrameDataset


@dataclass
class ModelArguments:
    model_name_or_path: str
    num_frames: int
    verb: bool


@dataclass
class DataArguments:
    train_frames_dir: str
    val_frames_dir: str
    train_annotation_file: str = None  # type: ignore
    val_annotation_file: str = None  # type: ignore


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True

    processor = transformers.VideoMAEImageProcessor.from_pretrained(
        model_args.model_name_or_path
    )
    if "shortest_edge" in processor.size:
        height = width = processor.size["shortest_edge"]
    else:
        height = processor.size["height"]
        width = processor.size["width"]

    def data_filter(item: dict) -> bool:
        return (
            item["structured_verb"] not in {"", "[other]"}
            and item["structured_noun"] != ""
        )

    train_data = FrameDataset(
        data_args.train_frames_dir,
        annotation_file=data_args.train_annotation_file,
        data_filter=data_filter,
        transform=ApplyTransformToKey(
            "video",
            transform=Compose(
                [
                    UniformTemporalSubsample(model_args.num_frames),
                    Lambda(lambda x: x * processor.rescale_factor),
                    Normalize(processor.image_mean, processor.image_std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop((height, width)),
                    RandomHorizontalFlip(),
                    Permute((1, 0, 2, 3)),
                ]
            ),
        ),
    )
    val_data = FrameDataset(
        data_args.val_frames_dir,
        annotation_file=data_args.val_annotation_file,
        data_filter=data_filter,
        transform=ApplyTransformToKey(
            "video",
            transform=Compose(
                [
                    UniformTemporalSubsample(model_args.num_frames),
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

    # Can't use train_data and val_data here since their transform functions fail b/c
    # we set return_frames to False
    tmp_train_data = FrameDataset(
        data_args.train_frames_dir,
        annotation_file=data_args.train_annotation_file,
        data_filter=data_filter,
        return_frames=False,
    )
    tmp_val_data = FrameDataset(
        data_args.val_frames_dir,
        annotation_file=data_args.val_annotation_file,
        data_filter=data_filter,
        return_frames=False,
    )
    label_key = "structured_verb" if model_args.verb else "structured_noun"
    labels = sorted({item[label_key] for item in iter(tmp_train_data + tmp_val_data)})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    model = transformers.VideoMAEForVideoClassification.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        label2id=label2id,
        id2label=id2label,
        num_frames=model_args.num_frames,
    )

    def compute_metrics(eval_pred):
        return {
            "f1": multiclass_f1_score(
                torch.tensor(eval_pred.predictions).argmax(dim=1),
                torch.tensor(eval_pred.label_ids),
                len(labels),
            ).item()
        }

    def collate_fn(examples: list[dict[str, Any]]):
        pixel_values = torch.stack([example["video"] for example in examples])
        labels = torch.tensor([label2id[example[label_key]] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
