import argparse
import json
import random

import wandb
from tqdm import tqdm

from eilev.data.frame import FrameInterleavedDataset

parser = argparse.ArgumentParser()
parser.add_argument("--in_context_frames_dir", required=True)
parser.add_argument("--in_context_annotation_file")
parser.add_argument("--eval_frames_dir", required=True)
parser.add_argument("--eval_annotation_file")
parser.add_argument("--num_shot", required=True, type=int)
parser.add_argument("--output_prefix", required=True)
parser.add_argument("--verb_noun_ratio", required=True, type=float)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--log_to_wandb", action="store_true")
args = parser.parse_args()

run_name = f"{args.output_prefix}-{args.num_shot}-shot"
artifact = None
if args.log_to_wandb:
    wandb.init(name=run_name, config=args)  # type: ignore
    artifact = wandb.Artifact(run_name, "dataset")

random.seed(args.random_seed)

dataset = FrameInterleavedDataset(
    args.eval_frames_dir,
    annotation_file=args.eval_annotation_file,
    in_context_example_frames_dir=args.in_context_frames_dir,
    in_context_example_annotation_file=args.in_context_annotation_file,
    num_in_context_examples_per_sample=args.num_shot,
    verb_noun_ratio=args.verb_noun_ratio,
    return_frames=False,
)
fname = f"{run_name}.jsonl"
with open(fname, "w") as f:
    for datapoint in tqdm(dataset):
        frame_paths = [item["frame_path"] for item in datapoint["items"]]
        f.write(
            json.dumps({"context": frame_paths[:-1], "query": frame_paths[-1]}) + "\n"
        )
if artifact is not None:
    artifact.add_file(fname)
    wandb.log_artifact(artifact)
