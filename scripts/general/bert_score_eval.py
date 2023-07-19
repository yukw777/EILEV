import argparse
import csv
from pprint import pprint

import numpy as np
import wandb
from torchmetrics.text.bert import BERTScore


def eval(rows: list[dict], bert_score: BERTScore) -> None:
    preds = [row["generated"] for row in rows]
    target = [row["ground_truth"] for row in rows]
    eval_results = bert_score(preds, target)
    table = wandb.Table(
        columns=[
            "frame_path",
            "video_uid",
            "clip_index",
            "generated",
            "ground_truth",
            "bert_score_f1",
            "bert_score_precision",
            "bert_score_recall",
        ]
    )
    for row, f1, precision, recall in zip(
        rows,
        eval_results["f1"],
        eval_results["precision"],
        eval_results["recall"],
    ):
        table.add_data(
            row["frame_path"],
            row["video_uid"],
            row["clip_index"],
            row["generated"],
            row["ground_truth"],
            f1,
            precision,
            recall,
        )

    log_dict = {
        "bert_score_table": table,
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
        rows = list(reader)

    bert_score = BERTScore(
        device=args.device,
        batch_size=args.batch_size,
        num_threads=0,
        rescale_with_baseline=True,
        verbose=True,
    )

    eval(rows, bert_score)
