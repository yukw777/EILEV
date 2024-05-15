import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("narrated_action_files", nargs="+")
parser.add_argument("split_output_path")
parser.add_argument("train_val_split", type=float)
parser.add_argument("--num_common_action", type=int)
parser.add_argument("--common_percent", type=float)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.num_common_action is not None and args.common_percent is not None:
    print("Only one of --num_common_action and --common_percent can be set.")
    exit(1)

random.seed(args.seed)

# read all the narrated actions
narrated_actions: list[dict] = []


def read_narrated_actions(csv_file: str, narrated_actions: list[dict]) -> None:
    with open(csv_file, newline="") as f:
        csvreader = csv.DictReader(f)
        narrated_actions.extend(csvreader)


for narrated_action_file in args.narrated_action_files:
    read_narrated_actions(narrated_action_file, narrated_actions)

print(f"Total # of narrated actions: {len(narrated_actions)}")
print()


def split_common_rare(
    counter: Counter, num_common_action: int | None, common_percent: float | None
) -> tuple[list[str], list[str]]:
    items, counts = zip(*[(item, count) for item, count in counter.most_common()])

    if num_common_action is not None:
        cut_off = num_common_action
    else:
        assert common_percent is not None
        cumulative_sum = np.cumsum(counts)
        cut_off = (
            np.where(cumulative_sum >= common_percent * cumulative_sum[-1])[0][0] + 1
        )

    return items[:cut_off], items[cut_off:]


verb_noun_pair_counter = Counter(
    (narrated_action["structured_verb"], narrated_action["structured_noun"])
    for narrated_action in narrated_actions
    if narrated_action["structured_verb"] and narrated_action["structured_noun"]
)
common_pairs, rare_pairs = split_common_rare(
    verb_noun_pair_counter, args.num_common_action, args.common_percent
)

print(f"Total # of common pairs: {len(common_pairs)}")
print(f"Top 10 common pairs: {common_pairs[:10]}")
print()
print(f"Total # of rare pairs: {len(rare_pairs)}")
print(f"Top 10 rare pairs: {rare_pairs[:10]}")
print()

train_val: list[dict] = []
test: list[dict] = []
common_pairs_set = set(common_pairs)
rare_pairs_set = set(rare_pairs)
for narrated_action in narrated_actions:
    pair = (
        narrated_action["structured_verb"],
        narrated_action["structured_noun"],
    )
    if pair in common_pairs_set:
        # narrated actions with common pairs become the train/val set
        train_val.append(narrated_action)
    elif pair in rare_pairs_set:
        # narrated actions with rare pairs become the test set
        test.append(narrated_action)

random.shuffle(train_val)

# split train val
split_index = round(len(train_val) * args.train_val_split)
train = train_val[:split_index]
val = train_val[split_index:]

print(f"# of train: {len(train)}")
print(f"# of val: {len(val)}")
print(f"# of test: {len(test)}")
print(
    "# of unused narrated actions: "
    f"{len(narrated_actions) - len(train)-len(val)-len(test)}"
)

output_path = Path(args.split_output_path)
output_path.mkdir(parents=True, exist_ok=True)
columns = list(train[0].keys())
with open(output_path / "train.csv", "w", newline="") as train_f:
    csvwriter = csv.DictWriter(train_f, columns)
    csvwriter.writeheader()
    for narrated_action in train:
        csvwriter.writerow(narrated_action)

with open(output_path / "val.csv", "w", newline="") as val_f:
    csvwriter = csv.DictWriter(val_f, columns)
    csvwriter.writeheader()
    for narrated_action in val:
        csvwriter.writerow(narrated_action)

with open(output_path / "test.csv", "w", newline="") as test_f:
    csvwriter = csv.DictWriter(test_f, columns)
    csvwriter.writeheader()
    for narrated_action in test:
        csvwriter.writerow(narrated_action)
