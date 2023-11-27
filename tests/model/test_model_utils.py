from unittest.mock import Mock

import pytest
import torch
from transformers import BatchEncoding

from eilev.model.utils import process


@pytest.mark.parametrize(
    "batch,time,height,width,resize",
    [(None, 8, 1280, 720, 244), (3, 8, 1280, 720, 244)],
)
def test_process(batch, time, height, width, resize):
    processor = Mock(
        side_effect=[
            BatchEncoding(
                data={
                    "pixel_values": torch.empty(
                        time if batch is None else batch * time, 3, resize, resize
                    )
                }
            )
        ]
    )
    assert (
        process(
            processor,
            video=torch.empty(3, time, height, width)
            if batch is None
            else torch.empty(batch, 3, time, height, width),
        ).pixel_values.size()
        == (1, 3, time, resize, resize)
        if batch is None
        else (batch, 3, time, resize, resize)
    )
