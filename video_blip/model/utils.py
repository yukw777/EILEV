import torch
from transformers import BatchEncoding, Blip2Processor


def process(
    processor: Blip2Processor,
    video: torch.Tensor | None = None,
    text: str | list[str] | None = None,
) -> BatchEncoding:
    """Process videos and texts for VideoBLIP.

    :param video: a tensor of shape (batch, channel, time, height, width) or
        (channel, time, height, width)
    """
    if video is not None:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        batch, channel, time, _, _ = video.size()
        video = video.permute(0, 2, 1, 3, 4).flatten(end_dim=1)
    inputs = processor(images=video, text=text, return_tensors="pt")
    if video is not None:
        _, _, height, weight = inputs.pixel_values.size()
        inputs["pixel_values"] = inputs.pixel_values.view(
            batch, time, channel, height, weight
        ).permute(0, 2, 1, 3, 4)
    return inputs
