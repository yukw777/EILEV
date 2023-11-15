import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("combined_file")
parser.add_argument("metric")
parser.add_argument("metric_threshold", type=float)
args = parser.parse_args()

with open(args.combined_file) as f:
    for i, line in enumerate(f):
        instance = json.loads(line)
        if (
            float(instance[args.metric]) >= args.metric_threshold
            and instance["structured_verb"] != ""
            and instance["structured_noun"] != ""
        ):
            print(f"Instance {i+1}: {instance['frame_path']}")
            print(f"Generated: {instance['generated']}")
            print(f"Ground-truth: {instance['ground_truth']}")
            print("==========================================")
            print()
