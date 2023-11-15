import argparse
import csv
import json
from pathlib import Path

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--evaluated_generated_narrations", required=True)
parser.add_argument("--evaluated_generated_narrations_annotations", required=True)
parser.add_argument("--in_context_examples", required=True)
parser.add_argument("--in_context_example_annotations", required=True)
parser.add_argument("--log_to_wandb", action="store_true")
args = parser.parse_args()

run_name = f"{Path(args.evaluated_generated_narrations).stem}-with-in-context"
artifact = None
if args.log_to_wandb:
    wandb.init(name=run_name, config=args)  # type: ignore
    artifact = wandb.Artifact(run_name, "dataset")

# read in the evaluated generated narrations
evaluated_generated_narrations = {}
with open(args.evaluated_generated_narrations, newline="") as f:
    csvreader = csv.DictReader(f)
    for row in csvreader:
        evaluated_generated_narrations[row["frame_path"]] = row

# read in the evaluated generated narrations annotations
evaluated_generated_narrations_annotations = {}
with open(args.evaluated_generated_narrations_annotations, newline="") as f:
    csvreader = csv.DictReader(f)
    for row in csvreader:
        evaluated_generated_narrations_annotations[row["frame_path"]] = row

# sanity check to make sure we have annotations for all evaluated generated narrations
assert (
    len(
        set(evaluated_generated_narrations.keys())
        - set(evaluated_generated_narrations_annotations.keys())
    )
    == 0
)

# read in the in-context examples
in_context_examples = {}
with open(args.in_context_examples) as f:
    for line in f:
        data = json.loads(line)
        in_context_examples[data["query"]] = data

# sanity check to make sure evaluated generated narrations and
# in-context examples match up
assert evaluated_generated_narrations.keys() == in_context_examples.keys()

# read in the in-context example annotations
in_context_example_annotations = {}
with open(args.in_context_example_annotations, newline="") as f:
    csvreader = csv.DictReader(f)
    for row in csvreader:
        in_context_example_annotations[row["frame_path"]] = row

# sanity check to make sure we have annotations for all in-context examples
assert (
    len(
        {ex for _, data in in_context_examples.items() for ex in data["context"]}
        - set(in_context_example_annotations.keys())
    )
    == 0
)

# combine them
fname = f"{run_name}.jsonl"
with open(fname, "w") as f:
    for frame_path, narration in evaluated_generated_narrations.items():
        # first add the structured verb and noun for the query
        narration["structured_verb"] = evaluated_generated_narrations_annotations[
            frame_path
        ]["structured_verb"]
        narration["structured_noun"] = evaluated_generated_narrations_annotations[
            frame_path
        ]["structured_noun"]

        # now add information about the context
        narration["context"] = [
            in_context_example_annotations[ctx_frame_path]
            for ctx_frame_path in in_context_examples[frame_path]["context"]
        ]
        f.write(json.dumps(narration) + "\n")

if artifact is not None:
    artifact.add_file(fname)
    wandb.log_artifact(artifact)
