from unittest.mock import Mock, patch

import pytest
import torch
from transformers import BatchEncoding

from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels,
    generate_input_ids_and_labels_from_interleaved,
)


@pytest.mark.parametrize(
    "narration_text,cleaned",
    [
        ("#C C drops a plate", "The camera wearer drops a plate"),
        ("#C C drops a plate ", "The camera wearer drops a plate"),
        ("#c C drops a plate", "The camera wearer drops a plate"),
        ("#C c drops a plate", "The camera wearer drops a plate"),
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


@pytest.mark.parametrize(
    "decoder_only_lm,tokenizer,num_texts,num_videos,text_video_map,expected",
    [
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [99, 1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                ],
                eos_token_id=100,
            ),
            1,
            1,
            [[0]],
            {
                "input_ids": torch.tensor([99, 1, 2, 3, 4, 4, 3, 2, 1, 100]),
                "labels": torch.tensor([-100, -100, -100, -100, -100, 4, 3, 2, 1, 100]),
                "video_causal_mask": torch.ones(10, 1).long(),
            },
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                ]
            ),
            1,
            1,
            [[0]],
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 100]),
                "labels": torch.tensor([4, 3, 2, 1, 100]),
                "video_causal_mask": torch.ones(5, 1).long(),
            },
        ),
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [99, 1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                ],
                eos_token_id=100,
            ),
            1,
            2,
            [[0, 1]],
            {
                "input_ids": torch.tensor([99, 1, 2, 3, 4, 4, 3, 2, 1, 100]),
                "labels": torch.tensor([-100, -100, -100, -100, -100, 4, 3, 2, 1, 100]),
                "video_causal_mask": torch.ones(10, 2).long(),
            },
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                ]
            ),
            1,
            2,
            [[0, 1]],
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 100]),
                "labels": torch.tensor([4, 3, 2, 1, 100]),
                "video_causal_mask": torch.ones(5, 2).long(),
            },
        ),
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [99, 1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                    BatchEncoding(data={"input_ids": [99, 5, 6, 7, 8]}),
                    BatchEncoding(data={"input_ids": [8, 7, 6, 5]}),
                    BatchEncoding(data={"input_ids": [99, 3, 4, 5, 6]}),
                    BatchEncoding(data={"input_ids": [6, 5, 4, 3]}),
                ],
                eos_token_id=100,
            ),
            3,
            3,
            [[0], [1], [2]],
            {
                "input_ids": torch.tensor(
                    [
                        99,
                        1,
                        2,
                        3,
                        4,
                        4,
                        3,
                        2,
                        1,
                        5,
                        6,
                        7,
                        8,
                        8,
                        7,
                        6,
                        5,
                        3,
                        4,
                        5,
                        6,
                        6,
                        5,
                        4,
                        3,
                        100,
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
                        6,
                        5,
                        4,
                        3,
                        100,
                    ]
                ),
                "video_causal_mask": torch.tensor(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                    ]
                ),
            },
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                    BatchEncoding(data={"input_ids": [5, 6, 7, 8, 100]}),
                    BatchEncoding(data={"input_ids": [8, 7, 6, 5, 100]}),
                    BatchEncoding(data={"input_ids": [3, 4, 5, 6, 100]}),
                    BatchEncoding(data={"input_ids": [6, 5, 4, 3, 100]}),
                ],
            ),
            3,
            3,
            [[0], [1], [2]],
            {
                "input_ids": torch.tensor(
                    [
                        1,
                        2,
                        3,
                        4,
                        4,
                        3,
                        2,
                        1,
                        5,
                        6,
                        7,
                        8,
                        8,
                        7,
                        6,
                        5,
                        3,
                        4,
                        5,
                        6,
                        100,
                    ]
                ),
                "labels": torch.tensor([6, 5, 4, 3, 100]),
                "video_causal_mask": torch.tensor(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                    ]
                ),
            },
        ),
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [99, 1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                    BatchEncoding(data={"input_ids": [99, 5, 6, 7, 8]}),
                    BatchEncoding(data={"input_ids": [8, 7, 6, 5]}),
                    BatchEncoding(data={"input_ids": [99, 3, 4, 5, 6]}),
                    BatchEncoding(data={"input_ids": [6, 5, 4, 3]}),
                ],
                eos_token_id=100,
            ),
            3,
            5,
            [[0, 1], [2], [3, 4]],
            {
                "input_ids": torch.tensor(
                    [
                        99,
                        1,
                        2,
                        3,
                        4,
                        4,
                        3,
                        2,
                        1,
                        5,
                        6,
                        7,
                        8,
                        8,
                        7,
                        6,
                        5,
                        3,
                        4,
                        5,
                        6,
                        6,
                        5,
                        4,
                        3,
                        100,
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
                        6,
                        5,
                        4,
                        3,
                        100,
                    ]
                ),
                "video_causal_mask": torch.tensor(
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                    ]
                ),
            },
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                    BatchEncoding(data={"input_ids": [5, 6, 7, 8, 100]}),
                    BatchEncoding(data={"input_ids": [8, 7, 6, 5, 100]}),
                    BatchEncoding(data={"input_ids": [3, 4, 5, 6, 100]}),
                    BatchEncoding(data={"input_ids": [6, 5, 4, 3, 100]}),
                ],
            ),
            3,
            5,
            [[0, 1], [2], [3, 4]],
            {
                "input_ids": torch.tensor(
                    [
                        1,
                        2,
                        3,
                        4,
                        4,
                        3,
                        2,
                        1,
                        5,
                        6,
                        7,
                        8,
                        8,
                        7,
                        6,
                        5,
                        3,
                        4,
                        5,
                        6,
                        100,
                    ]
                ),
                "labels": torch.tensor([6, 5, 4, 3, 100]),
                "video_causal_mask": torch.tensor(
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                    ]
                ),
            },
        ),
    ],
)
def test_generate_input_ids_and_labels_from_interleaved(
    decoder_only_lm: bool,
    tokenizer,
    num_texts: int,
    num_videos: int,
    text_video_map: list[list[int]],
    expected: dict[str, torch.Tensor],
) -> None:
    results = generate_input_ids_and_labels_from_interleaved(
        tokenizer,
        [""] * num_texts,
        [""] * num_texts,
        num_videos,
        text_video_map,
        decoder_only_lm,
    )
    assert results.keys() == expected.keys()
    assert results["input_ids"].equal(expected["input_ids"])
    assert results["labels"].equal(expected["labels"])
    assert results["video_causal_mask"].equal(expected["video_causal_mask"])


@pytest.mark.parametrize(
    "datapoints,expected",
    [
        (
            [
                {
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                }
            ],
            torch.tensor([[[1, 0], [1, 0], [0, 1]]]),
        ),
        (
            [
                {
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                },
                {
                    "video_causal_mask": torch.tensor(
                        [
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 1],
                        ]
                    ),
                },
                {
                    "video_causal_mask": torch.tensor(
                        [
                            [1, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                        ]
                    ),
                },
            ],
            torch.tensor(
                [
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                    ],
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
        ),
    ],
)
def test_data_collator_for_interleaved_video_seq2seq(datapoints, expected):
    with patch(
        "video_blip.data.utils.DataCollatorForVideoSeq2Seq.__call__",
        return_value=BatchEncoding(),
    ):
        collator = DataCollatorForInterleavedVideoSeq2Seq(Mock())
        collated = collator(datapoints)
        assert collated["video_causal_mask"].equal(expected)
