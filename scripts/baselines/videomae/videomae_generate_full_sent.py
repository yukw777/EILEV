import argparse
import csv
from pathlib import Path

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from eilev.data.utils import clean_narration_text, generate_chunks

parser = argparse.ArgumentParser()
parser.add_argument("prediction_file")
parser.add_argument("--device", default="cpu")
parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--print_full_sent", action="store_true")
parser.add_argument("--log_to_wandb", action="store_true")
args = parser.parse_args()

table: wandb.Table | None
if args.log_to_wandb:
    run_name = Path(args.prediction_file).stem
    run = wandb.init(name=run_name, config=args)  # type: ignore
    table = wandb.Table(
        columns=[
            "frame_path",
            "video_uid",
            "clip_index",
            "predicted_structured_verb",
            "ground_truth_structured_verb",
            "predicted_structured_noun",
            "ground_truth_structured_noun",
            "generated",
            "ground_truth",
        ]
    )
else:
    table = None
    run = None

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
model = LlamaForCausalLM.from_pretrained(args.model).to(args.device)

prompt_template = """
Use the given verb and noun classes to generate a complete sentence using "the camera wearer" as the subject.

Verb class: open
Noun class: mold
Sentence: The camera wearer opens a mold.

Verb class: put_(place,_leave,_drop)
Noun class: bag_(bag,_grocery,_nylon,_polythene,_pouch,_sachet,_sack,_suitcase)
Sentence: The camera wearer drops a bag.

Verb class: hold_(support,_grip,_grasp)
Noun class: rod_(dipstick,_rod,_rod_metal,_shaft)
Sentence: The camera wearer grasps a rod.

Verb class: %s
Noun class: %s
Sentence:"""  # noqa: E501

with open(args.prediction_file) as pred_f:
    rows = [row for row in csv.DictReader(pred_f)]
batches = list(generate_chunks(rows, args.batch_size))
period_token_id = tokenizer.convert_tokens_to_ids(".")
for batch in tqdm(batches):
    prompts = [
        prompt_template
        % (row["predicted_structured_verb"], row["predicted_structured_noun"])
        for row in batch
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
    outputs = model.generate(**inputs, max_new_tokens=64, eos_token_id=period_token_id)
    for row, all_generated, prompt in zip(
        batch, tokenizer.batch_decode(outputs, skip_special_tokens=True), prompts
    ):
        generated = all_generated[len(prompt) :].strip()
        if args.print_full_sent:
            print(f"Generated: {generated}")
        if table is not None:
            table.add_data(
                row["frame_path"],
                row["video_uid"],
                row["clip_index"],
                row["predicted_structured_verb"],
                row["ground_truth_structured_verb"],
                row["predicted_structured_noun"],
                row["ground_truth_structured_noun"],
                generated,
                clean_narration_text(row["ground_truth_narration_text"]),
            )
if table is not None and run is not None:
    run.log({"generated": table})
