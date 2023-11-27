from unittest.mock import Mock, patch

import pytest

from eilev.data.frame import FrameInterleavedDataset


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
@patch("eilev.data.frame.random.sample", new=lambda p, k: sorted(p)[:k])
def test_ego4d_fho_main_frame_interleaved_dataset(data, expected):
    with patch("eilev.data.frame.FrameDataset") as mock_parent_dataset:
        mock_parent_dataset_instance = Mock(data=data)
        mock_parent_dataset_instance.__len__ = Mock(return_value=len(data))
        mock_parent_dataset_instance.__getitem__ = Mock(
            side_effect=lambda i: {"pixel_values": "torch.tensor", **data[i]}
        )
        mock_parent_dataset.return_value = mock_parent_dataset_instance
        dataset = FrameInterleavedDataset("hi")
        assert [d for d in dataset] == expected


@pytest.mark.parametrize(
    "data,in_context_data,expected",
    [
        (
            [
                {
                    "id": 0,
                    "structured_verb": "verb0",
                    "structured_noun": "noun0",
                },
                {
                    "id": 1,
                    "structured_verb": "verb1",
                    "structured_noun": "noun1",
                },
            ],
            [
                {
                    "id": -1,
                    "structured_verb": "verb0",
                    "structured_noun": "noun0",
                },
                {
                    "id": -2,
                    "structured_verb": "verb0",
                    "structured_noun": "noun1",
                },
                {
                    "id": -3,
                    "structured_verb": "verb0",
                    "structured_noun": "noun2",
                },
                {
                    "id": -4,
                    "structured_verb": "verb0",
                    "structured_noun": "noun3",
                },
                {
                    "id": -5,
                    "structured_verb": "verb1",
                    "structured_noun": "noun0",
                },
                {
                    "id": -6,
                    "structured_verb": "verb1",
                    "structured_noun": "noun1",
                },
                {
                    "id": -7,
                    "structured_verb": "verb1",
                    "structured_noun": "noun2",
                },
                {
                    "id": -8,
                    "structured_verb": "verb1",
                    "structured_noun": "noun3",
                },
            ],
            [
                {
                    "items": [
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -3,
                            "structured_verb": "verb0",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -4,
                            "structured_verb": "verb0",
                            "structured_noun": "noun3",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -5,
                            "structured_verb": "verb1",
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
                            "pixel_values": "in-context torch.tensor",
                            "id": -2,
                            "structured_verb": "verb0",
                            "structured_noun": "noun1",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -5,
                            "structured_verb": "verb1",
                            "structured_noun": "noun0",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -7,
                            "structured_verb": "verb1",
                            "structured_noun": "noun2",
                        },
                        {
                            "pixel_values": "in-context torch.tensor",
                            "id": -8,
                            "structured_verb": "verb1",
                            "structured_noun": "noun3",
                        },
                        {
                            "pixel_values": "torch.tensor",
                            "id": 1,
                            "structured_verb": "verb1",
                            "structured_noun": "noun1",
                        },
                    ]
                },
            ],
        ),
    ],
)
@patch("eilev.data.frame.random.sample", new=lambda p, k: sorted(p)[:k])
def test_ego4d_fho_main_frame_interleaved_dataset_in_context_dataset(
    data, in_context_data, expected
):
    with patch("eilev.data.frame.FrameDataset") as mock_parent_dataset:
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

        def mock_parent_dataset_init(frames_dir, **kwargs):
            if frames_dir == "data":
                return mock_parent_dataset_instance
            return mock_in_context_dataset_instance

        mock_parent_dataset.side_effect = mock_parent_dataset_init
        dataset = FrameInterleavedDataset(
            "data", in_context_example_frames_dir="in-context-data"
        )
        assert [d for d in dataset] == expected
