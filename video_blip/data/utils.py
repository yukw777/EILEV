import re
from collections.abc import Iterable
from typing import TypeVar

import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer

C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)


class DataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        collated["pixel_values"] = pixel_values
        return collated


def clean_narration_text(narration_text: str) -> str:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    return re.sub(C_REGEX, "The camera wearer", cleaned)


def generate_input_ids_and_labels(
    tokenizer: PreTrainedTokenizer, prompt: str, text: str, decoder_only_lm: bool
) -> BatchEncoding:
    """Generate input ids and labels from the given prompt and text. If
    decoder_only_lm is True, the input and label texts are the same, but label
    tokens that correspond to the prompt are masked with -100. If
    decoder_only_lm is False, the input corresponds to the prompt and the label
    to the text.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompt: prompt for the LLM
    :param text: text for the LLM to generate based on the prompt
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results
    """
    if decoder_only_lm:
        # tokenize prompt first
        prompt_tokens = tokenizer(prompt, return_attention_mask=False).input_ids

        # tokenize the narration and append eos
        preprocessed = tokenizer(
            " " + text,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        preprocessed["input_ids"].append(tokenizer.eos_token_id)

        # join tokenized prompt and narration text
        preprocessed["input_ids"] = prompt_tokens + preprocessed["input_ids"]
        preprocessed["input_ids"] = torch.tensor(preprocessed.input_ids)

        # for decoder only LMs, labels are same as input_ids, but we mask
        # tokens for the prompt
        preprocessed["labels"] = preprocessed["input_ids"].clone()
        preprocessed["labels"][: len(prompt_tokens)] = -100
    else:
        # eos is automatically appended by the tokenizer
        # we don't use return_tensors='pt' here b/c it automatically batchifies things
        # which we don't want
        preprocessed = tokenizer(prompt, return_attention_mask=False)
        preprocessed["input_ids"] = torch.tensor(preprocessed["input_ids"])
        preprocessed["labels"] = torch.tensor(
            tokenizer(text, return_attention_mask=False).input_ids
        )

    return preprocessed


def generate_input_ids_and_labels_from_interleaved(
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    texts: list[str],
    num_videos: int,
    text_video_map: list[list[int]],
    decoder_only_lm: bool,
) -> dict[str, torch.Tensor]:
    """Generate input ids and labels from the given interleaved video/text data
    point. We treat the last prompt and text as labels and the rest as the
    context. `text_video_map` specifies which videos are the last preceding
    videos for a given text, and is used to generate `video_causal_mask`.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompts: list of prompt for the LLM
    :param texts: list of texts for the LLM to generate based on the prompt
    :param num_videos: number of videos in this interleaved data point
    :param text_video_map: map between texts and their last preceding videos.
        each text can have zero, one or more (if there are multiple consecutive videos
        preceding the text) last preceding videos.
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results including `input_ids`, `labels` and
        `video_causal_mask`.
        `input_ids` is a tensor of shape (num_tokens),
        `labels` is a tensor of shape (num_tokens),
        `video_causal_mask` is a tensor of shape (num_tokens, num_videos)
    """
    assert len(prompts) == len(texts)
    assert len(text_video_map) == len(texts)

    processed_texts: list[BatchEncoding] = []
    for i, (prompt, text) in enumerate(zip(prompts, texts)):
        processed = generate_input_ids_and_labels(
            tokenizer, prompt, text, decoder_only_lm
        )
        if decoder_only_lm:
            # decoder only lm
            if i != len(texts) - 1:
                # if not last, set all labels to -100 as it's part of the context.
                processed["labels"] = torch.full_like(processed["labels"], -100)
            if i != 0:
                # if not first, remove bos
                processed["input_ids"] = processed["input_ids"][1:]
                processed["labels"] = processed["labels"][1:]
        else:
            if i != len(texts) - 1:
                # if not last, append labels to input_ids after removing eos from it
                # as it's part of the context.
                processed["input_ids"] = torch.cat(
                    [processed["input_ids"][:-1], processed["labels"]]
                )
                processed["labels"] = torch.empty(0)

        if i != len(texts) - 1:
            # if not last, remove eos
            processed["input_ids"] = processed["input_ids"][:-1]
            processed["labels"] = processed["labels"][:-1]
        processed_texts.append(processed)

    video_causal_mask = torch.zeros(
        sum(processed.input_ids.size(0) for processed in processed_texts),
        num_videos,
        dtype=torch.long,
    )
    start_token_index = 0
    for i, video_indices in enumerate(text_video_map):
        processed = processed_texts[i]
        end_token_index = start_token_index + processed.input_ids.size(0)
        video_causal_mask[start_token_index:end_token_index].index_fill_(
            1, torch.tensor(video_indices), 1
        )
        start_token_index = end_token_index

    return {
        "input_ids": torch.cat([processed.input_ids for processed in processed_texts]),
        "labels": torch.cat([processed.labels for processed in processed_texts]),
        "video_causal_mask": video_causal_mask,
    }


T = TypeVar("T")


def generate_chunks(list_to_chunk: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i : i + chunk_size]
