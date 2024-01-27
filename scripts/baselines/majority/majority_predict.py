import argparse
from collections import Counter
from pathlib import Path

import spacy
import wandb
from tqdm import tqdm

from eilev.data.frame import FrameInterleavedPresampledDataset
from eilev.data.utils import clean_narration_text

parser = argparse.ArgumentParser()
parser.add_argument("--eval_frames_dir", required=True)
parser.add_argument("--eval_annotation_file")
parser.add_argument("--in_context_query_map_file", required=True)
parser.add_argument("--in_context_example_frames_dir", required=True)
parser.add_argument("--in_context_example_annotation_file")
parser.add_argument("--print_predictions", action="store_true")
parser.add_argument("--num_eval_datapoints", default=None, type=int)
parser.add_argument("--log_to_wandb", action="store_true")
args = parser.parse_args()

table: wandb.Table | None
if args.log_to_wandb:
    run_name = f"majority-class-{Path(args.in_context_query_map_file).stem}"
    run = wandb.init(name=run_name, config=args)  # type: ignore
    table = wandb.Table(
        columns=[
            "frame_path",
            "video_uid",
            "clip_index",
            "predicted_verb",
            "ground_truth_structured_verb",
            "predicted_noun",
            "ground_truth_structured_noun",
            "ground_truth_narration_text",
        ]
    )
else:
    table = None
    run = None

dataset = FrameInterleavedPresampledDataset(
    args.eval_frames_dir,
    args.in_context_query_map_file,
    args.in_context_example_frames_dir,
    annotation_file=args.eval_annotation_file,
    in_context_example_annotation_file=args.in_context_example_annotation_file,
    return_frames=False,
)
nlp = spacy.load("en_core_web_sm")

for datapoint in tqdm(dataset, desc="Predicting"):
    in_context_examples = datapoint["items"][:-1]
    query = datapoint["items"][-1]
    narrations = [
        clean_narration_text(example["narration_text"])
        for example in in_context_examples
    ]
    verb_counter: Counter[str] = Counter()
    noun_counter: Counter[str] = Counter()
    for doc in nlp.pipe(narrations, disable=["ner"]):
        for token in doc:
            if token.dep_ == "ROOT":
                verb_counter[token.lemma_] += 1
                for child in token.children:
                    if child.dep_ == "dobj":
                        noun_counter[child.lemma_] += 1
    predicted_verb = ""
    if len(verb_counter) != 0:
        predicted_verb = verb_counter.most_common(1)[0][0]
    predicted_noun = ""
    if len(noun_counter) != 0:
        predicted_noun = noun_counter.most_common(1)[0][0]
    if args.print_predictions:
        print(
            f"Predicted verb: {predicted_verb}, "
            f"Ground-truth verb: {query['structured_verb']}"
        )
        print(
            f"Predicted noun: {predicted_noun}, "
            f"Ground-truth noun: {query['structured_noun']}"
        )
        print(f"Ground-truth narration text: {query['narration_text']}")
    if table is not None:
        table.add_data(
            query["frame_path"],
            query["video_uid"],
            query["clip_index"],
            predicted_verb,
            query["structured_verb"],
            predicted_noun,
            query["structured_noun"],
            query["narration_text"],
        )

if table is not None and run is not None:
    run.log({"predictions": table})
