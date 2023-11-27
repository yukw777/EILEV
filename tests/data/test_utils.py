from fractions import Fraction
from unittest.mock import Mock, patch

import pytest
import torch
from pytorchvideo.data.clip_sampling import ClipInfo
from transformers import BatchEncoding, Blip2Processor

from eilev.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    NarratedActionClipSampler,
    clean_narration_text,
    generate_input_ids_and_labels,
    generate_input_ids_and_labels_from_interleaved,
    parse_timestamp,
)


@pytest.mark.parametrize(
    "narration_text,cleaned",
    [
        ("#C C drops a plate", "The camera wearer drops a plate."),
        ("#C C drops a plate ", "The camera wearer drops a plate."),
        ("#c C drops a plate", "The camera wearer drops a plate."),
        ("#C c drops a plate", "The camera wearer drops a plate."),
        (
            "#C  C adjusts the screw in the machine with the screwdriver. #Unsure."
            "<|eos|>",
            "The camera wearer adjusts the screw in the machine with the screwdriver.",
        ),
        (
            "#C C drops #unsure on the countertop<|eos|>",
            "The camera wearer drops something on the countertop.",
        ),
        (
            "#C C touches his face with his right hand. #Unsure.\n<|eos|>",
            "The camera wearer touches his face with his right hand.",
        ),
        (
            "#C C fixes the wire into the piston. #unsure.<|eos|>",
            "The camera wearer fixes the wire into the piston.",
        ),
        (
            "#C C fixes the wire into the piston. #unsure<|eos|>",
            "The camera wearer fixes the wire into the piston.",
        ),
        (
            "#C C pours cooking pots #Unsure on the        <|eos|>",
            "The camera wearer pours cooking pots something on the.",
        ),
    ],
)
def test_clean_narration_text(narration_text: str, cleaned: str) -> None:
    assert clean_narration_text(narration_text) == cleaned


@pytest.mark.parametrize(
    "decoder_only_lm,tokenizer,expected",
    [
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                ],
                eos_token_id=100,
            ),
            BatchEncoding(
                data={
                    "input_ids": torch.tensor([1, 2, 3, 4, 4, 3, 2, 1, 100]),
                    "labels": torch.tensor([-100, -100, -100, -100, 4, 3, 2, 1, 100]),
                }
            ),
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                ]
            ),
            BatchEncoding(
                data={
                    "input_ids": torch.tensor([1, 2, 3, 4, 100]),
                    "labels": torch.tensor([4, 3, 2, 1, 100]),
                }
            ),
        ),
    ],
)
def test_generate_input_ids_and_labels(
    decoder_only_lm: bool, tokenizer, expected: BatchEncoding
) -> None:
    results = generate_input_ids_and_labels(tokenizer, "", "", decoder_only_lm)
    assert results.keys() == expected.keys()
    assert results.input_ids.equal(expected.input_ids)
    assert results.labels.equal(expected.labels)


@pytest.fixture
def decoder_only_processor():
    return Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")


@pytest.fixture
def seq2seq_processor():
    return Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")


@pytest.mark.parametrize(
    "prompts,text,num_query_tokens,expected",
    [
        (
            [("A prompt", 1)],
            "A text",
            2,
            {
                "input_ids": torch.tensor(
                    [2, 1, 1, 50118, 250, 14302, 83, 2788, 50118, 2]
                ),
                "labels": torch.tensor(
                    [-100, -100, -100, -100, -100, -100, 83, 2788, 50118, 2]
                ),
                "video_input_mask": torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            },
        ),
        (
            [("A prompt", 1)],
            None,
            4,
            {
                "input_ids": torch.tensor([2, 1, 1, 1, 1, 50118, 250, 14302]),
                "labels": torch.tensor(
                    [-100, -100, -100, -100, -100, -100, -100, -100]
                ),
                "video_input_mask": torch.tensor([0, 1, 1, 1, 1, 0, 0, 0]),
            },
        ),
        (
            [("A prompt", 2)],
            "A text",
            2,
            {
                "input_ids": torch.tensor(
                    [2, 1, 1, 50118, 1, 1, 50118, 250, 14302, 83, 2788, 50118, 2]
                ),
                "labels": torch.tensor(
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        83,
                        2788,
                        50118,
                        2,
                    ]
                ),
                "video_input_mask": torch.tensor(
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                ),
            },
        ),
        (
            [("A prompt", 2)],
            None,
            4,
            {
                "input_ids": torch.tensor(
                    [2, 1, 1, 1, 1, 50118, 1, 1, 1, 1, 50118, 250, 14302]
                ),
                "labels": torch.tensor(
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                    ]
                ),
                "video_input_mask": torch.tensor(
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
                ),
            },
        ),
        (
            [
                ("Prompt 1 Text 1", 2),
                ("Prompt 2 Text 2", 1),
                ("Prompt 3", 2),
            ],
            "Text 3",
            2,
            {
                "input_ids": torch.tensor(
                    [
                        2,
                        1,
                        1,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        112,
                        14159,
                        112,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        132,
                        14159,
                        132,
                        50118,
                        1,
                        1,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        155,
                        14159,
                        155,
                        50118,
                        2,
                    ]
                ),
                "labels": torch.tensor(
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        14159,
                        155,
                        50118,
                        2,
                    ]
                ),
                "video_input_mask": torch.tensor(
                    [
                        0,
                        1,
                        1,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        ),
        (
            [
                ("Prompt 1 Text 1", 2),
                ("Prompt 2 Text 2", 1),
                ("Prompt 3", 2),
            ],
            None,
            2,
            {
                "input_ids": torch.tensor(
                    [
                        2,
                        1,
                        1,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        112,
                        14159,
                        112,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        132,
                        14159,
                        132,
                        50118,
                        1,
                        1,
                        50118,
                        1,
                        1,
                        50118,
                        35396,
                        3320,
                        155,
                    ]
                ),
                "labels": torch.tensor(
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                    ]
                ),
                "video_input_mask": torch.tensor(
                    [
                        0,
                        1,
                        1,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        ),
    ],
)
def test_generate_input_ids_and_labels_from_interleaved_decoder_only(
    decoder_only_processor: Blip2Processor,
    prompts: list[tuple[str, int]],
    text: str | None,
    num_query_tokens: int,
    expected: dict[str, torch.Tensor],
) -> None:
    results = generate_input_ids_and_labels_from_interleaved(
        decoder_only_processor.tokenizer, prompts, text, num_query_tokens, True
    )
    assert results.keys() == expected.keys()
    assert results["input_ids"].equal(expected["input_ids"])
    assert results["labels"].equal(expected["labels"])
    assert results["video_input_mask"].equal(expected["video_input_mask"])


@pytest.mark.parametrize(
    "prompts,text,num_query_tokens,expected",
    [
        (
            [("A prompt", 1)],
            "A text",
            2,
            {
                "input_ids": torch.tensor([0, 0, 3, 71, 9005, 1]),
                "labels": torch.tensor([71, 1499, 1]),
                "video_input_mask": torch.tensor([1, 1, 0, 0, 0, 0]),
            },
        ),
        (
            [("A prompt", 1)],
            None,
            4,
            {
                "input_ids": torch.tensor([0, 0, 0, 0, 3, 71, 9005, 1]),
                "labels": torch.tensor([]),
                "video_input_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
            },
        ),
        (
            [("A prompt", 2)],
            "A text",
            2,
            {
                "input_ids": torch.tensor([0, 0, 3, 0, 0, 3, 71, 9005, 1]),
                "labels": torch.tensor([71, 1499, 1]),
                "video_input_mask": torch.tensor([1, 1, 0, 1, 1, 0, 0, 0, 0]),
            },
        ),
        (
            [("A prompt", 2)],
            None,
            4,
            {
                "input_ids": torch.tensor([0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 71, 9005, 1]),
                "labels": torch.tensor([]),
                "video_input_mask": torch.tensor(
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
                ),
            },
        ),
        (
            [("Prompt 1 Text 1", 1), ("Prompt 2 Text 2", 1), ("Prompt 3", 1)],
            "Text 3",
            2,
            {
                "input_ids": torch.tensor(
                    [
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        209,
                        5027,
                        209,
                        3,
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        204,
                        5027,
                        204,
                        3,
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        220,
                        1,
                    ]
                ),
                "labels": torch.tensor([5027, 220, 1]),
                "video_input_mask": torch.tensor(
                    [
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        ),
        (
            [("Prompt 1 Text 1", 1), ("Prompt 2 Text 2", 1), ("Prompt 3 Text 3", 1)],
            None,
            2,
            {
                "input_ids": torch.tensor(
                    [
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        209,
                        5027,
                        209,
                        3,
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        204,
                        5027,
                        204,
                        3,
                        0,
                        0,
                        3,
                        749,
                        1167,
                        17,
                        220,
                        5027,
                        220,
                        1,
                    ]
                ),
                "labels": torch.tensor([]),
                "video_input_mask": torch.tensor(
                    [
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        ),
    ],
)
def test_generate_input_ids_and_labels_from_interleaved_seq2seq(
    seq2seq_processor: Blip2Processor,
    prompts: list[tuple[str, int]],
    text: str | None,
    num_query_tokens: int,
    expected: dict[str, torch.Tensor],
) -> None:
    results = generate_input_ids_and_labels_from_interleaved(
        seq2seq_processor.tokenizer, prompts, text, num_query_tokens, False
    )
    assert results.keys() == expected.keys()
    assert results["input_ids"].equal(expected["input_ids"])
    assert results["labels"].equal(expected["labels"])
    assert results["video_input_mask"].equal(expected["video_input_mask"])


@pytest.mark.parametrize(
    "datapoints,padding_side,pad_to_multiple_of,expected_pixel_values,"
    "expected_video_input_mask",
    [
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                }
            ],
            "left",
            None,
            torch.ones(2, 1, 1, 1, 1),
            torch.tensor([[0, 1, 1, 0]]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                }
            ],
            "right",
            None,
            torch.ones(2, 1, 1, 1, 1),
            torch.tensor([[0, 1, 1, 0]]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                }
            ],
            "left",
            8,
            torch.ones(2, 1, 1, 1, 1),
            torch.tensor([[0] * 4 + [0, 1, 1, 0]]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                }
            ],
            "right",
            8,
            torch.ones(2, 1, 1, 1, 1),
            torch.tensor([[0, 1, 1, 0] + [0] * 4]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
                    "input_ids": torch.ones(6).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1]),
                },
                {
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
                    "input_ids": torch.ones(7).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1, 0]),
                },
            ],
            "left",
            None,
            torch.ones(9, 1, 1, 1, 1),
            torch.tensor(
                [
                    [0] * 3 + [0, 1, 1, 0],
                    [0] + [0, 1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1, 0],
                ]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
                    "input_ids": torch.ones(6).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1]),
                },
                {
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
                    "input_ids": torch.ones(7).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1, 0]),
                },
            ],
            "right",
            None,
            torch.ones(9, 1, 1, 1, 1),
            torch.tensor(
                [
                    [0, 1, 1, 0] + [0] * 3,
                    [0, 1, 1, 0, 1, 1] + [0],
                    [0, 1, 1, 0, 1, 1, 0],
                ]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
                    "input_ids": torch.ones(6).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1]),
                },
                {
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
                    "input_ids": torch.ones(7).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1, 0]),
                },
            ],
            "left",
            8,
            torch.ones(9, 1, 1, 1, 1),
            torch.tensor(
                [
                    [0] * 4 + [0, 1, 1, 0],
                    [0] * 2 + [0, 1, 1, 0, 1, 1],
                    [0] + [0, 1, 1, 0, 1, 1, 0],
                ]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "input_ids": torch.ones(4).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
                    "input_ids": torch.ones(6).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1]),
                },
                {
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
                    "input_ids": torch.ones(7).long(),
                    "video_input_mask": torch.tensor([0, 1, 1, 0, 1, 1, 0]),
                },
            ],
            "right",
            8,
            torch.ones(9, 1, 1, 1, 1),
            torch.tensor(
                [
                    [0, 1, 1, 0] + [0] * 4,
                    [0, 1, 1, 0, 1, 1] + [0] * 2,
                    [0, 1, 1, 0, 1, 1, 0] + [0],
                ]
            ),
        ),
    ],
)
def test_data_collator_for_interleaved_video_seq2seq(
    decoder_only_processor,
    datapoints,
    padding_side,
    pad_to_multiple_of,
    expected_pixel_values,
    expected_video_input_mask,
):
    decoder_only_processor.tokenizer.padding_side = padding_side
    collator = DataCollatorForInterleavedVideoSeq2Seq(
        decoder_only_processor.tokenizer, pad_to_multiple_of=pad_to_multiple_of
    )
    collated = collator(datapoints)
    assert collated["pixel_values"].equal(expected_pixel_values)
    assert collated["video_input_mask"].equal(expected_video_input_mask)


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        ("00:00:00.560", 0.56),
        ("00:00:49.15", 49.15),
        ("00:06:50.039", 410.039),
        ("02:06:50.039", 7610.039),
    ],
)
def test_parse_timestamps(timestamp, expected):
    assert parse_timestamp(timestamp) == expected


def reverse(x: list[int]) -> None:
    x.reverse()


@patch("eilev.data.frame.random.shuffle", new=reverse)
def test_narrated_action_clip_sampler_random() -> None:
    clip_sampler = NarratedActionClipSampler(True)
    video_duration_1 = 12
    annotation_1 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 2},
            {"narration_timestamp_sec": 6},
            {"narration_timestamp_sec": 10},
        ]
    }
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )

    annotation_2 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 3},
            {"narration_timestamp_sec": 7},
            {"narration_timestamp_sec": 10},
        ]
    }
    video_duration_2 = 14
    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(6), Fraction(14), 2, 0, False
    )
    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(3), Fraction(11), 1, 0, False
    )
    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )


def test_narrated_action_clip_sampler() -> None:
    clip_sampler = NarratedActionClipSampler(False)
    video_duration_1 = 12
    annotation_1 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 2},
            {"narration_timestamp_sec": 6},
            {"narration_timestamp_sec": 10},
        ]
    }
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, True
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, video_duration_1, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, True
    )

    annotation_2 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 3},
            {"narration_timestamp_sec": 7},
            {"narration_timestamp_sec": 10},
        ]
    }
    video_duration_2 = 14

    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(3), Fraction(11), 1, 0, False
    )
    assert clip_sampler(0, video_duration_2, annotation_2) == ClipInfo(
        Fraction(6), Fraction(14), 2, 0, True
    )
