import argparse
import csv

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from eilev.data.utils import generate_chunks

parser = argparse.ArgumentParser()
parser.add_argument("annotation")
parser.add_argument("annotation_with_std_sent")
parser.add_argument("--device", default="cpu")
parser.add_argument("--dtype")
parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
}
model = LlamaForCausalLM.from_pretrained(args.model).to(
    args.device, dtype_dict[args.dtype]
)

prompt_template = """Use the verb and noun to generate a sentence using "the camera wearer" as the subject.

Verb: cut
Noun: plant
Generated: The camera wearer cuts the plant.

Verb: repair
Noun: car
Generated: The camera wearer repairs the car.

Verb: move
Noun: tablet
Generated: The camera wearer moves the tablet.

Verb: %s
Noun: %s
Generated:"""  # noqa: E501

rows: list[dict] = []
with open(args.annotation) as ann_f:
    for row in csv.DictReader(ann_f):
        if row["structured_verb"] == "" or row["structured_noun"] == "":
            continue
        row["verb"] = row["structured_verb"].split("_", 1)[0]
        row["noun"] = row["structured_noun"].split("_", 1)[0]
        rows.append(row)

batches = list(generate_chunks(rows, args.batch_size))
newline_token_id = tokenizer.convert_tokens_to_ids("<0x0A>")
with open(args.annotation_with_std_sent, "w", newline="") as ann_full_sent_f:
    csv_writer = csv.DictWriter(
        ann_full_sent_f, list(k for k in rows[0].keys() if k not in {"verb", "noun"})
    )
    csv_writer.writeheader()
    for batch in tqdm(batches):
        prompts = [prompt_template % (row["verb"], row["noun"]) for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        outputs = model.generate(
            **inputs, max_new_tokens=64, eos_token_id=newline_token_id
        )
        for row, full_sent_narration, prompt in zip(
            batch, tokenizer.batch_decode(outputs, skip_special_tokens=True), prompts
        ):
            narration_text = full_sent_narration[len(prompt) :].strip()
            narration_text = narration_text.split(".", maxsplit=1)[0] + "."
            row["narration_text"] = narration_text
            del row["verb"]
            del row["noun"]
        csv_writer.writerows(batch)
