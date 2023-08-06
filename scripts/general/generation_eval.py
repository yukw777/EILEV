import argparse
import csv
from pprint import pprint

import numpy as np
import torch
import wandb
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torchmetrics.text import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore


def calc_sts_bi_encoder(
    preds: list[str], target: list[str], batch_size: int, device: str
) -> list[float]:
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    encoded_preds = model.encode(
        preds,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    encoded_target = model.encode(
        target,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    results = util.pairwise_cos_sim(encoded_preds, encoded_target).tolist()
    del model
    torch.cuda.empty_cache()
    return results


def calc_sts_cross_encoder(
    preds: list[str], target: list[str], batch_size: int, device: str
) -> list[float]:
    model = CrossEncoder("cross-encoder/stsb-roberta-large", device=device)
    results = model.predict(
        list(zip(preds, target)), batch_size=batch_size, show_progress_bar=True
    ).tolist()
    del model
    torch.cuda.empty_cache()
    return results


def calc_rouge(preds: list[str], target: list[str]) -> dict[str, float]:
    rouge = ROUGEScore(rouge_keys="rougeL")
    return {k: v.item() for k, v in rouge(preds, target).items()}


def calc_bleu(preds: list[str], target: list[str]) -> float:
    bleu = BLEUScore()
    return bleu(preds, [[t] for t in target]).item()


def calc_bertscore(
    preds: list[str], target: list[str], batch_size: int, device: str
) -> dict[str, list[float]]:
    bert_score = BERTScore(
        batch_size=batch_size,
        device=device,
        num_threads=0,
        rescale_with_baseline=True,
        verbose=True,
    )

    results = bert_score(preds, target)
    del bert_score
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_narration_file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bertscore_batch_size", default=64, type=int)
    parser.add_argument("--sts_bi_encoder_batch_size", default=64, type=int)
    parser.add_argument("--sts_cross_encoder_batch_size", default=64, type=int)
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
    sts_bi_encoder_results = calc_sts_bi_encoder(
        preds, target, args.sts_bi_encoder_batch_size, args.device
    )
    sts_cross_encoder_results = calc_sts_cross_encoder(
        preds, target, args.sts_cross_encoder_batch_size, args.device
    )

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
            "sts_bi_encoder_cos_sim",
            "sts_cross_encoder_score",
        ]
    )
    for (
        row,
        bertscore_f1,
        bertscore_precision,
        bertscore_recall,
        sts_bi_encoder_cos_sim,
        sts_cross_encoder_score,
    ) in zip(
        rows,
        bertscore_results["f1"],
        bertscore_results["precision"],
        bertscore_results["recall"],
        sts_bi_encoder_results,
        sts_cross_encoder_results,
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
            sts_bi_encoder_cos_sim,
            sts_cross_encoder_score,
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
        "sts_bi_encoder_mean_cos_sim": np.array(sts_bi_encoder_results).mean(),
        "sts_cross_encoder_mean_score": np.array(sts_cross_encoder_results).mean(),
    }
    pprint(log_dict)
    wandb.log(log_dict)
