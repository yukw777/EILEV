import argparse
import csv
import os
import random
from collections import Counter

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("train_narrated_actions")
parser.add_argument("val_narrated_actions")
parser.add_argument("test_narrated_actions")
parser.add_argument("split_output_path")
parser.add_argument("common_percent", type=float)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# read all the narrated actions
narrated_actions: list[dict] = []


def read_narrated_actions(csv_file: str, narrated_actions: list[dict]) -> None:
    with open(csv_file, newline="") as f:
        csvreader = csv.DictReader(f)
        narrated_actions.extend(csvreader)


read_narrated_actions(args.train_narrated_actions, narrated_actions)
read_narrated_actions(args.val_narrated_actions, narrated_actions)
read_narrated_actions(args.test_narrated_actions, narrated_actions)

print(f"Total # of narrated actions: {len(narrated_actions)}")
print()


def split_common_rare(
    counter: Counter, common_percent: float
) -> tuple[list[str], list[str]]:
    items, counts = zip(*[(item, count) for item, count in counter.most_common()])

    cumulative_sum = np.cumsum(counts)
    cut_off = np.where(cumulative_sum >= common_percent * cumulative_sum[-1])[0][0]

    return items[: cut_off + 1], items[cut_off + 1 :]


verb_noun_pair_counter = Counter(
    (narrated_action["structured_verb"], narrated_action["structured_noun"])
    for narrated_action in narrated_actions
    if narrated_action["structured_verb"] and narrated_action["structured_noun"]
)
common_pairs, rare_pairs = split_common_rare(
    verb_noun_pair_counter, args.common_percent
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
split_index = round(len(train_val) * args.common_percent)
train = train_val[:split_index]
val = train_val[split_index:]

print(f"# of train: {len(train)}")
print(f"# of val: {len(val)}")
print(f"# of test: {len(test)}")
print(
    "# of unused narrated actions: "
    f"{len(narrated_actions) - len(train)-len(val)-len(test)}"
)

columns = list(train[0].keys())
with open(
    os.path.join(args.split_output_path, "train.csv"), "w", newline=""
) as train_f:
    csvwriter = csv.DictWriter(train_f, columns)
    csvwriter.writeheader()
    for narrated_action in train:
        csvwriter.writerow(narrated_action)

with open(os.path.join(args.split_output_path, "val.csv"), "w", newline="") as val_f:
    csvwriter = csv.DictWriter(val_f, columns)
    csvwriter.writeheader()
    for narrated_action in val:
        csvwriter.writerow(narrated_action)

with open(os.path.join(args.split_output_path, "test.csv"), "w", newline="") as test_f:
    csvwriter = csv.DictWriter(test_f, columns)
    csvwriter.writeheader()
    for narrated_action in test:
        csvwriter.writerow(narrated_action)
