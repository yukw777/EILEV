from unittest.mock import Mock, patch

import pytest
import torch
from transformers import BatchEncoding, Blip2Processor

from video_blip.data.utils import (
    DataCollatorForInterleavedVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels,
    generate_input_ids_and_labels_from_interleaved,
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
def processor():
    return Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")


@pytest.mark.parametrize(
    "prompts_texts,num_videos,text_video_map,expected",
    [
        (
            [("A prompt", "A text")],
            1,
            [[0]],
            {
                "input_ids": torch.tensor([2, 250, 14302, 83, 2788, 50118, 2]),
                "labels": torch.tensor([-100, -100, -100, 83, 2788, 50118, 2]),
                "video_causal_mask": torch.ones(7, 1).long(),
            },
        ),
        (
            [("A prompt", None)],
            1,
            [[0]],
            {
                "input_ids": torch.tensor([2, 250, 14302]),
                "labels": torch.tensor([-100, -100, -100]),
                "video_causal_mask": torch.ones(3, 1).long(),
            },
        ),
        (
            [("A prompt", "A text")],
            2,
            [[0, 1]],
            {
                "input_ids": torch.tensor([2, 250, 14302, 83, 2788, 50118, 2]),
                "labels": torch.tensor([-100, -100, -100, 83, 2788, 50118, 2]),
                "video_causal_mask": torch.ones(7, 2).long(),
            },
        ),
        (
            [("A prompt", None)],
            2,
            [[0, 1]],
            {
                "input_ids": torch.tensor([2, 250, 14302]),
                "labels": torch.tensor([-100, -100, -100]),
                "video_causal_mask": torch.ones(3, 2).long(),
            },
        ),
        (
            [("Prompt 1", "Text 1"), ("Prompt 2", "Text 2"), ("Prompt 3", "Text 3")],
            3,
            [[0], [1], [2]],
            {
                "input_ids": torch.tensor(
                    [
                        2,
                        35396,
                        3320,
                        112,
                        14159,
                        112,
                        50118,
                        35396,
                        3320,
                        132,
                        14159,
                        132,
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
                        14159,
                        112,
                        50118,
                        -100,
                        -100,
                        -100,
                        14159,
                        132,
                        50118,
                        -100,
                        -100,
                        -100,
                        14159,
                        155,
                        50118,
                        2,
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
                    ]
                ),
            },
        ),
        (
            [("Prompt 1", "Text 1"), ("Prompt 2", "Text 2"), ("Prompt 3", "Text 3")],
            5,
            [[0, 1], [2], [3, 4]],
            {
                "input_ids": torch.tensor(
                    [
                        2,
                        35396,
                        3320,
                        112,
                        14159,
                        112,
                        50118,
                        35396,
                        3320,
                        132,
                        14159,
                        132,
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
                        14159,
                        112,
                        50118,
                        -100,
                        -100,
                        -100,
                        14159,
                        132,
                        50118,
                        -100,
                        -100,
                        -100,
                        14159,
                        155,
                        50118,
                        2,
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
                    ]
                ),
            },
        ),
        (
            [
                ("Prompt 1 Text 1", ""),
                ("Prompt 2 Text 2", ""),
                ("Prompt 3", None),
            ],
            5,
            [[0, 1], [2], [3, 4]],
            {
                "input_ids": torch.tensor(
                    [
                        2,
                        35396,
                        3320,
                        112,
                        14159,
                        112,
                        50118,
                        35396,
                        3320,
                        132,
                        14159,
                        132,
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
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
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
    processor: Blip2Processor,
    prompts_texts: list[tuple[str, str | None]],
    num_videos: int,
    text_video_map: list[list[int]],
    expected: dict[str, torch.Tensor],
) -> None:
    results = generate_input_ids_and_labels_from_interleaved(
        processor.tokenizer, prompts_texts, num_videos, text_video_map
    )
    assert results.keys() == expected.keys()
    assert results["input_ids"].equal(expected["input_ids"])
    assert results["labels"].equal(expected["labels"])
    assert results["video_causal_mask"].equal(expected["video_causal_mask"])


@pytest.mark.parametrize(
    "datapoints,attention_mask,padding_side,expected_pixel_values,"
    "expected_video_causal_mask",
    [
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                }
            ],
            torch.ones(1, 3).long(),
            "left",
            torch.ones(1, 2, 1, 1, 1, 1).long(),
            torch.tensor([[[1, 0], [1, 0], [0, 1]]]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                }
            ],
            torch.ones(1, 3).long(),
            "right",
            torch.ones(1, 2, 1, 1, 1, 1).long(),
            torch.tensor([[[1, 0], [1, 0], [0, 1]]]),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                }
            ],
            # pad to multiple of 8
            torch.tensor([[1] * 3 + [0] * 5]),
            "left",
            torch.ones(1, 2, 1, 1, 1, 1).long(),
            torch.tensor(
                [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [0, 1]]]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                }
            ],
            # pad to multiple of 8
            torch.tensor([[1] * 3 + [0] * 5]),
            "right",
            torch.ones(1, 2, 1, 1, 1, 1).long(),
            torch.tensor(
                [[[1, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
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
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
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
            torch.tensor([[1] * 3 + [0] * 4, [1] * 7, [1] * 6 + [0]]),
            "left",
            torch.stack(
                [
                    torch.concat(
                        [torch.ones(2, 1, 1, 1, 1), torch.zeros(2, 1, 1, 1, 1)]
                    ),
                    torch.concat(
                        [torch.ones(3, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1)]
                    ),
                    torch.ones(4, 1, 1, 1, 1),
                ]
            ),
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
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
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                    ],
                ]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
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
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
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
            torch.tensor([[0] * 4 + [1] * 3, [1] * 7, [0] + [1] * 6]),
            "right",
            torch.stack(
                [
                    torch.concat(
                        [torch.ones(2, 1, 1, 1, 1), torch.zeros(2, 1, 1, 1, 1)]
                    ),
                    torch.concat(
                        [torch.ones(3, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1)]
                    ),
                    torch.ones(4, 1, 1, 1, 1),
                ]
            ),
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
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
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
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
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
            # pad to multiple of 8
            torch.tensor([[1] * 3 + [0] * 5, [1] * 7 + [0], [1] * 6 + [0] * 2]),
            "left",
            torch.stack(
                [
                    torch.concat(
                        [torch.ones(2, 1, 1, 1, 1), torch.zeros(2, 1, 1, 1, 1)]
                    ),
                    torch.concat(
                        [torch.ones(3, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1)]
                    ),
                    torch.ones(4, 1, 1, 1, 1),
                ]
            ),
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                    ],
                ]
            ),
        ),
        (
            [
                {
                    "pixel_values": torch.ones(2, 1, 1, 1, 1),
                    "video_causal_mask": torch.tensor([[1, 0], [1, 0], [0, 1]]),
                },
                {
                    "pixel_values": torch.ones(3, 1, 1, 1, 1),
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
                    "pixel_values": torch.ones(4, 1, 1, 1, 1),
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
            # pad to multiple of 8
            torch.tensor([[0] * 5 + [1] * 3, [0] + [1] * 7, [0] * 2 + [1] * 6]),
            "right",
            torch.stack(
                [
                    torch.concat(
                        [torch.ones(2, 1, 1, 1, 1), torch.zeros(2, 1, 1, 1, 1)]
                    ),
                    torch.concat(
                        [torch.ones(3, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1)]
                    ),
                    torch.ones(4, 1, 1, 1, 1),
                ]
            ),
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
                        [0, 0, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
        ),
    ],
)
def test_data_collator_for_interleaved_video_seq2seq(
    datapoints,
    attention_mask,
    padding_side,
    expected_pixel_values,
    expected_video_causal_mask,
):
    def mock_call(features, return_tensors):
        return BatchEncoding(
            data={
                "attention_mask": attention_mask,
                "pixel_values": torch.stack(
                    [feature.pop("pixel_values") for feature in features]
                ),
            }
        )

    with patch(
        "video_blip.data.utils.DataCollatorForVideoSeq2Seq.__call__",
        side_effect=mock_call,
    ):
        collator = DataCollatorForInterleavedVideoSeq2Seq(
            Mock(padding_side=padding_side)
        )
        collated = collator(datapoints)
        assert collated["pixel_values"].equal(expected_pixel_values)
        assert collated["video_causal_mask"].equal(expected_video_causal_mask)
