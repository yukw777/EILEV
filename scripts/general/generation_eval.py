import argparse
import csv
from pprint import pprint

import numpy as np
import wandb
from torchmetrics.text import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore


def calc_rouge(preds: list[str], target: list[str]) -> dict[str, float]:
    rouge = ROUGEScore(rouge_keys="rougeL")
    return {k: v.item() for k, v in rouge(preds, target).items()}


def calc_bleu(preds: list[str], target: list[str]) -> float:
    bleu = BLEUScore()
    return bleu(preds, target).item()


def calc_bertscore(
    preds: list[str], target: list[str], batch_size: int, device: "str"
) -> dict[str, list[float]]:
    bert_score = BERTScore(
        batch_size=batch_size,
        device=device,
        num_threads=0,
        rescale_with_baseline=True,
        verbose=True,
    )

    return bert_score(preds, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_narration_file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bertscore_batch_size", default=64, type=int)
    args = parser.parse_args()
    wandb.init(config=args)  # type: ignore

    with open(args.gen_narration_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    preds = [row["generated"] for row in rows]
    target = [row["ground_truth"] for row in rows]

    bertscore_results = calc_bertscore(
        preds, target, args.bertscore_batch_size, args.device
    )
    bleu_score = calc_bleu(preds, target)
    rouge_results = calc_rouge(preds, target)

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
    for (
        row,
        bertscore_f1,
        bertscore_precision,
        bertscore_recall,
    ) in zip(
        rows,
        bertscore_results["f1"],
        bertscore_results["precision"],
        bertscore_results["recall"],
    ):
        table.add_data(
            row["frame_path"],
            row["video_uid"],
            row["clip_index"],
            row["generated"],
            row["ground_truth"],
            bertscore_f1,
            bertscore_precision,
            bertscore_recall,
        )

    log_dict = {
        "eval_results": table,
        "bert_score_mean_f1": np.array(bertscore_results["f1"]).mean(),
        "bert_score_mean_precision": np.array(bertscore_results["precision"]).mean(),
        "bert_score_mean_recall": np.array(bertscore_results["recall"]).mean(),
        "bleu_score": bleu_score,
        "rougeL_fmeasure": rouge_results["rougeL_fmeasure"],
        "rougeL_precision": rouge_results["rougeL_precision"],
        "rougeL_recall": rouge_results["rougeL_recall"],
    }
    pprint(log_dict)
    wandb.log(log_dict)
