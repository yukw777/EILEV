import argparse
import csv

from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from eilev.data.utils import generate_chunks

parser = argparse.ArgumentParser()
parser.add_argument("annotation")
parser.add_argument("annotation_with_full_sent")
parser.add_argument("--device", default="cpu")
parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
model = LlamaForCausalLM.from_pretrained(args.model).to(args.device)

prompt_template = """Fix a phrase into a complete sentence using "the camera wearer" as the subject.

Phrase: close drawer
Fixed: The camera wearer closes the drawer.

Phrase: add thyme to dough
Fixed: The camera wearer adds thyme to the dough.

Phrase: push fish cake into bowl
Fixed: The camera wearer pushes the fish cake into the bowl.

Phrase: %s
Fixed:"""  # noqa: E501

with open(args.annotation) as ann_f:
    rows = [row for row in csv.DictReader(ann_f)]
batches = list(generate_chunks(rows, args.batch_size))
newline_token_id = tokenizer.convert_tokens_to_ids("<0x0A>")
with open(args.annotation_with_full_sent, "w", newline="") as ann_full_sent_f:
    csv_writer = csv.DictWriter(
        ann_full_sent_f, list(rows[0].keys()) + ["full_sent_narration"]
    )
    csv_writer.writeheader()
    for batch in tqdm(batches):
        prompts = [prompt_template % row["narration"] for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        outputs = model.generate(
            **inputs, max_new_tokens=64, eos_token_id=newline_token_id
        )
        for row, full_sent_narration, prompt in zip(
            batch, tokenizer.batch_decode(outputs, skip_special_tokens=True), prompts
        ):
            row["full_sent_narration"] = full_sent_narration[len(prompt) :].strip()
        csv_writer.writerows(batch)
