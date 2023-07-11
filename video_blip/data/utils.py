import re
import string
from collections.abc import Iterable
from typing import TypeVar

import torch
import torch.nn.functional as F
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer

C_REGEX = re.compile(r"^\#C\s+C", re.IGNORECASE)
EOS_REGEX = re.compile(r"\<\|eos\|\>$", re.IGNORECASE)
UNSURE_END_REGEX = re.compile(r"#unsure\.?$", re.IGNORECASE)
UNSURE_MIDDLE_REGEX = re.compile(r"#unsure", re.IGNORECASE)


class DataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        collated["pixel_values"] = pixel_values
        return collated


class DataCollatorForInterleavedVideoSeq2Seq(DataCollatorForVideoSeq2Seq):
    def __call__(self, features, return_tensors=None):
        video_causal_mask_list = [
            feature.pop("video_causal_mask") for feature in features
        ]
        collated = super().__call__(features, return_tensors=return_tensors)
        # use the text token length calculated by super() as it handles
        # pad_to_multiple_of.
        max_text_token_len = collated["attention_mask"].size(1)
        max_video_token_len = max(mask.size(1) for mask in video_causal_mask_list)
        video_causal_mask = torch.stack(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_video_token_len - mask.size(1),
                        0,
                        max_text_token_len - mask.size(0),
                    ),
                )
                for mask in video_causal_mask_list
            ]
        )
        collated["video_causal_mask"] = video_causal_mask
        return collated


def clean_narration_text(narration_text: str) -> str:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    cleaned = re.sub(C_REGEX, "The camera wearer", cleaned).strip()

    # remove <|eos|>
    cleaned = re.sub(EOS_REGEX, "", cleaned).strip()

    # remove #unsure from the end
    cleaned = re.sub(UNSURE_END_REGEX, "", cleaned).strip()

    # replace #unsure in the middle with "something"
    cleaned = re.sub(UNSURE_MIDDLE_REGEX, "something", cleaned)

    if len(cleaned) == 0:
        return cleaned

    # if cleaned doesn't end with a punctuation, append a period
    if not cleaned[-1] in string.punctuation:
        cleaned += "."

    return cleaned


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
    prompts_texts: list[tuple[str, str | None]],
    num_videos: int,
    text_video_map: list[list[int]],
) -> dict[str, torch.Tensor]:
    """Generate input ids and labels from the given interleaved video/text data
    point. `text_video_map` specifies which videos are the last preceding
    videos for a given text, and is used to generate `video_causal_mask`. Note
    that this is for autoregressive language modeling, so only decoder only LMs
    are supported.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompts_texts: list of tuples of prompts and texts. texts are for the LLM to
        generate based on the prompts, and can be None if not needed. Note that
        if text is None, an eos token is not appended, which is useful for generation.
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
    assert len(text_video_map) == len(prompts_texts)

    input_ids: list[int] = []
    labels: list[int] = []
    video_causal_mask: list[torch.Tensor] = []
    for i, (prompt, text) in enumerate(prompts_texts):
        if text is None:
            text_tokens: list[int] = []
        elif text == "":
            # append a newline separator to the prompt since text is an empty string
            prompt += "\n"
            text_tokens = []
        else:
            # prepend a space to separate the text from the prompt
            text_tokens = tokenizer(
                " " + text + "\n", add_special_tokens=False
            ).input_ids
        prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        if i == 0:
            # if first prompt, prepend a bos token
            prompt_tokens = [tokenizer.bos_token_id] + prompt_tokens
        if i == len(prompts_texts) - 1 and text is not None:
            # if last text, append eos
            text_tokens.append(tokenizer.eos_token_id)

        input_ids.extend(prompt_tokens + text_tokens)
        labels.extend([-100] * len(prompt_tokens) + text_tokens)

        # build video_causal_mask
        video_indices = text_video_map[i]
        for _ in range(len(prompt_tokens + text_tokens)):
            video_causal_mask.append(
                torch.zeros(num_videos, dtype=torch.long).index_fill(
                    0, torch.tensor(video_indices, dtype=torch.long), 1
                )
            )

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "video_causal_mask": torch.stack(video_causal_mask),
    }


T = TypeVar("T")


def generate_chunks(list_to_chunk: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i : i + chunk_size]
