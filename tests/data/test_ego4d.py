from fractions import Fraction
from unittest.mock import Mock, patch

import pytest
from pytorchvideo.data.clip_sampling import ClipInfo

from video_blip.data.ego4d import (
    Ego4dFHOMainFrameInterleavedDataset,
    NarratedActionClipSampler,
)


def reverse(x: list[int]) -> None:
    x.reverse()


@patch("video_blip.data.ego4d.random.shuffle", new=reverse)
def test_narrated_action_clip_sampler_random() -> None:
    clip_sampler = NarratedActionClipSampler(True)
    annotation_1 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 2},
            {"narration_timestamp_sec": 6},
            {"narration_timestamp_sec": 10},
        ],
        "video_metadata": {"duration_sec": 12},
    }
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )

    annotation_2 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 3},
            {"narration_timestamp_sec": 7},
            {"narration_timestamp_sec": 10},
        ],
        "video_metadata": {"duration_sec": 14},
    }

    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(6), Fraction(14), 2, 0, False
    )
    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(3), Fraction(11), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, True
    )


def test_narrated_action_clip_sampler() -> None:
    clip_sampler = NarratedActionClipSampler(False)
    annotation_1 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 2},
            {"narration_timestamp_sec": 6},
            {"narration_timestamp_sec": 10},
        ],
        "video_metadata": {"duration_sec": 12},
    }
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, True
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(2), Fraction(10), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_1) == ClipInfo(
        Fraction(4), Fraction(12), 2, 0, True
    )

    annotation_2 = {
        "narrated_actions": [
            {"narration_timestamp_sec": 3},
            {"narration_timestamp_sec": 7},
            {"narration_timestamp_sec": 10},
        ],
        "video_metadata": {"duration_sec": 14},
    }

    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(0), Fraction(8), 0, 0, False
    )
    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(3), Fraction(11), 1, 0, False
    )
    assert clip_sampler(0, 0, annotation_2) == ClipInfo(
        Fraction(6), Fraction(14), 2, 0, True
    )


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            [
                {"id": 0, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 1, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 2, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 3, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 4, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 5, "structured_verb": "verb1", "structured_noun": "noun1"},
                {"id": 6, "structured_verb": "verb1", "structured_noun": "noun1"},
                {"id": 7, "structured_verb": "verb1", "structured_noun": "noun1"},
                {"id": 8, "structured_verb": "verb1", "structured_noun": "noun1"},
                {"id": 9, "structured_verb": "verb1", "structured_noun": "noun1"},
            ],
            [
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 6,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 9,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 9,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 6,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 6,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 9,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 6,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 9,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 6,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 9,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
            ],
        ),
        (
            [
                {"id": 0, "structured_verb": "verb0", "structured_noun": "noun0"},
                {"id": 1, "structured_verb": "verb0", "structured_noun": "noun1"},
                {"id": 2, "structured_verb": "verb1", "structured_noun": "noun2"},
                {"id": 3, "structured_verb": "verb1", "structured_noun": "noun0"},
                {"id": 4, "structured_verb": "verb2", "structured_noun": "noun1"},
            ],
            [
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb2",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb2",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb2",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb2",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb0",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 2,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 3,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 4,
                            "structured_verb": "verb2",
                            "structured_noun": "noun1",
                        },
                    ]
                },
            ],
        ),
    ],
)
@patch("video_blip.data.ego4d.random.sample", new=lambda p, k: sorted(p)[:k])
def test_ego4d_fho_main_frame_interleaved_dataset(data, expected):
    with patch("video_blip.data.ego4d.Ego4dFHOMainFrameDataset") as mock_parent_dataset:
        mock_parent_dataset_instance = Mock(data=data)
        mock_parent_dataset_instance.__len__ = Mock(return_value=len(data))
        mock_parent_dataset_instance.__getitem__ = Mock(
            side_effect=lambda i: {"pixel_values": "torch.tensor", **data[i]}
        )
        mock_parent_dataset.return_value = mock_parent_dataset_instance
        dataset = Ego4dFHOMainFrameInterleavedDataset("hi")
        assert [d for d in dataset] == expected


@pytest.mark.parametrize(
    "data,in_context_data,expected",
    [
        (
            [
                {
                    "id": 0,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
                {
                    "id": 1,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
            ],
            [
                {
                    "id": -1,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
                {
                    "id": -2,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
                {
                    "id": -3,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
                {
                    "id": -4,
                    "structured_verb": "verb",
                    "structured_noun": "noun",
                },
            ],
            [
                {
                    "items": [
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -1,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -2,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -3,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -4,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 0,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                    ]
                },
                {
                    "items": [
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -1,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -2,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -3,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -4,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb",
                            "structured_noun": "noun",
                        },
                    ]
                },
            ],
        ),
    ],
)
@patch("video_blip.data.ego4d.random.sample", new=lambda p, k: sorted(p)[:k])
def test_ego4d_fho_main_frame_interleaved_dataset_in_context_dataset(
    data, in_context_data, expected
):
    with patch("video_blip.data.ego4d.Ego4dFHOMainFrameDataset") as mock_parent_dataset:
        mock_parent_dataset_instance = Mock(data=data)
        mock_parent_dataset_instance.__len__ = Mock(return_value=len(data))
        mock_parent_dataset_instance.__getitem__ = Mock(
            side_effect=lambda i: {"pixel_values": "torch.tensor", **data[i]}
        )
        mock_in_context_dataset_instance = Mock(data=in_context_data)
        mock_in_context_dataset_instance.__len__ = Mock(
            return_value=len(in_context_data)
        )
        mock_in_context_dataset_instance.__getitem__ = Mock(
            side_effect=lambda i: {
                "pixel_values": "in-context torch.tensor",
                **in_context_data[i],
            }
        )

        def mock_parent_dataset_init(path):
            if path == "data":
                return mock_parent_dataset_instance
            return mock_in_context_dataset_instance

        mock_parent_dataset.side_effect = mock_parent_dataset_init
        dataset = Ego4dFHOMainFrameInterleavedDataset(
            "data", in_context_example_narrated_actions_dir="in-context-data"
        )
        assert len(dataset) == len(expected)
        for a, b in zip(dataset, expected):
            assert a == b
