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
        if all("pixel_values" in feature for feature in features):
            pixel_values = torch.stack(
                [feature.pop("pixel_values") for feature in features]
            )
        else:
            # in some cases, we don't have pixel values, e.g.,
            # in-context learning evaluation
            pixel_values = None
        collated = super().__call__(features, return_tensors=return_tensors)
        if pixel_values is not None:
            collated["pixel_values"] = pixel_values
        return collated


class DataCollatorForInterleavedVideoSeq2Seq(DataCollatorForVideoSeq2Seq):
    def __call__(self, features, return_tensors=None):
        max_num_vids = max(feature["pixel_values"].size(0) for feature in features)
        video_causal_mask_list = []
        for feature in features:
            num_videos_to_pad = max_num_vids - feature["pixel_values"].size(0)
            feature["pixel_values"] = F.pad(
                feature["pixel_values"],
                (0, 0, 0, 0, 0, 0, 0, 0, 0, num_videos_to_pad),
            )
            video_causal_mask_list.append(
                F.pad(feature.pop("video_causal_mask"), (0, num_videos_to_pad))
            )

        collated = super().__call__(features, return_tensors=return_tensors)
        # use the text token length calculated by super() as it handles
        # pad_to_multiple_of.
        max_text_token_len = collated["attention_mask"].size(1)
        max_video_token_len = max(mask.size(1) for mask in video_causal_mask_list)
        collated["video_causal_mask"] = torch.stack(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_video_token_len - mask.size(1),
                        0,
                        max_text_token_len - mask.size(0),
                    )
                    if self.tokenizer.padding_side == "right"
                    else (
                        0,
                        max_video_token_len - mask.size(1),
                        max_text_token_len - mask.size(0),
                        0,
                    ),
                )
                for mask in video_causal_mask_list
            ]
        )
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
    prompts: list[tuple[str, int]],
    text: str | None,
    num_query_tokens: int,
    decoder_only_lm: bool,
) -> dict[str, torch.Tensor]:
    """Generate input ids and labels from the given interleaved video/text data
    point. `text_video_map` specifies which videos are the last preceding
    videos for a given text, and is used to generate `video_input_mask`.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompts: list of prompts, each with the number of videos
    :param text: optional text to be completed by LLM
    :param num_query_tokens: number of qformer query tokens
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results including `input_ids`, `labels` and
        `video_input_mask`.
        `input_ids` is a tensor of shape (num_tokens),
        `labels` is a tensor of shape (num_tokens),
        `video_input_mask` is a tensor of shape (num_tokens)
    """
    input_ids: list[int] = []
    labels: list[int] = []
    video_input_mask: list[int] = []
    # NOTE: FLAN tokenizer treats all whitespaces the same
    newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    if decoder_only_lm:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                )
                labels.extend([-100] * (num_query_tokens + 1))
                video_input_mask.extend([1] * num_query_tokens + [0])
            if i == 0:
                # if first text, start with a bos token
                input_ids = [tokenizer.bos_token_id] + input_ids
                labels = [-100] + labels
                video_input_mask = [0] + video_input_mask
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            input_ids.extend(prompt_tokens)
            video_input_mask.extend([0] * len(prompt_tokens))
            labels.extend([-100] * len(prompt_tokens))
        if text is not None:
            # prepend a space to separate the text from the prompt
            text_tokens = tokenizer(
                " " + text + "\n", add_special_tokens=False
            ).input_ids + [tokenizer.eos_token_id]
            input_ids.extend(text_tokens)
            video_input_mask.extend([0] * len(text_tokens))
            labels.extend(text_tokens)
    else:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                )
                video_input_mask.extend([1] * num_query_tokens + [0])
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            if i == len(prompts) - 1:
                # if last prompt, add eos token
                prompt_tokens.append(tokenizer.eos_token_id)
            input_ids.extend(prompt_tokens)
            video_input_mask.extend([0] * len(prompt_tokens))
        if text is not None:
            labels.extend(tokenizer(text).input_ids)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "video_input_mask": torch.tensor(video_input_mask),
    }


T = TypeVar("T")


def generate_chunks(list_to_chunk: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i : i + chunk_size]
