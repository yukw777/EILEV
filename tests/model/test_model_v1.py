import pytest
import torch
from transformers import Blip2VisionConfig

from eilev.model.v1 import VideoBlipVisionModel


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize(
    "config",
    [
        Blip2VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=8,
        ),
        Blip2VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=4,
            num_attention_heads=8,
            patch_size=12,
        ),
    ],
)
def test_v1_video_blip_vision_model_forward(
    config: Blip2VisionConfig,
    batch: int,
    time: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoBlipVisionModel(config)
    outputs = model(
        pixel_values=torch.rand(
            # channel is pretty much always 3
            batch,
            3,
            time,
            config.image_size,
            config.image_size,
        ),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    last_hidden_state, pooler_output, hidden_states, attentions = outputs
    # divide the image into non-overlapping patches, flatten them out,
    # then add a cls token.
    num_tokens = (config.image_size // config.patch_size) ** 2 + 1
    assert last_hidden_state.size() == (batch, time * num_tokens, config.hidden_size)
    assert pooler_output.size() == (batch, time, config.hidden_size)

    if output_attentions:
        assert len(attentions) == config.num_hidden_layers
        for attn in attentions:
            assert attn.size() == (
                batch,
                time,
                config.num_attention_heads,
                num_tokens,
                num_tokens,
            )
    else:
        assert attentions is None

    if output_hidden_states:
        # num_hidden_layers + 1 for embeddings
        assert len(hidden_states) == config.num_hidden_layers + 1
        for hidden in hidden_states:
            assert hidden.size() == (batch, time * num_tokens, config.hidden_size)
    else:
        assert hidden_states is None
