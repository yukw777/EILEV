from fractions import Fraction
from unittest.mock import patch

from pytorchvideo.data.clip_sampling import ClipInfo

from video_blip.data.ego4d import NarratedActionClipSampler


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
