import argparse
import csv
from pprint import pprint

import numpy as np
import wandb
from torchmetrics.text.bert import BERTScore


def eval(preds: list[str], target: list[str], bert_score: BERTScore) -> None:
    eval_results = bert_score(preds, target)
    log_dict = {
        "bert_score_mean_f1": np.array(eval_results["f1"]).mean(),
        "bert_score_mean_precision": np.array(eval_results["precision"]).mean(),
        "bert_score_mean_recall": np.array(eval_results["recall"]).mean(),
    }
    pprint(log_dict)
    wandb.log(log_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_narration_file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    wandb.init(config=args)  # type: ignore

    with open(args.gen_narration_file, newline="") as f:
        reader = csv.DictReader(f)
        preds: list[str] = []
        target: list[str] = []
        for row in reader:
            preds.append(row["generated"])
            target.append(row["ground_truth"])

    bert_score = BERTScore(
        device=args.device,
        batch_size=args.batch_size,
        num_threads=0,
        rescale_with_baseline=True,
        verbose=True,
    )

    eval(preds, target, bert_score)
