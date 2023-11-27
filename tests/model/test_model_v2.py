import pytest
import torch
from transformers import Blip2Config, Blip2VisionConfig

from eilev.model.v2 import VideoBlipForConditionalGeneration, VideoBlipVisionModel


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [1, 5])
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
def test_v2_video_blip_vision_model_forward(
    config: Blip2VisionConfig,
    num_videos: int,
    time: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoBlipVisionModel(config)
    outputs = model(
        pixel_values=torch.rand(
            num_videos,
            # channel is pretty much always 3
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
    assert last_hidden_state.size() == (
        num_videos,
        time * num_tokens,
        config.hidden_size,
    )
    assert pooler_output.size() == (num_videos, time, config.hidden_size)

    if output_attentions:
        assert len(attentions) == config.num_hidden_layers
        for attn in attentions:
            assert attn.size() == (
                num_videos,
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
            assert hidden.size() == (num_videos, time * num_tokens, config.hidden_size)
    else:
        assert hidden_states is None


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [None, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "config",
    [
        Blip2Config(
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "projection_dim": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 8,
            },
            qformer_config={
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            text_config={
                "model_type": "opt",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "ffn_dim": 16,
                "num_attention_heads": 2,
            },
            num_query_tokens=4,
        ),
        Blip2Config(
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "projection_dim": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 8,
            },
            qformer_config={
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            text_config={
                "model_type": "t5",
                "d_model": 8,
                "d_kv": 4,
                "d_ff": 16,
                "num_layers": 2,
                "num_heads": 2,
                "decoder_start_token_id": 0,
            },
            num_query_tokens=4,
        ),
    ],
)
def test_v2_video_blip_for_cond_gen(
    config: Blip2Config,
    batch: int,
    num_videos: int | None,
    time: int,
    seq_len: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoBlipForConditionalGeneration(config)
    outputs = model(
        torch.ones(batch, seq_len).long(),
        attention_mask=torch.ones(batch, seq_len).long(),
        pixel_values=torch.rand(
            num_videos,
            # channel is pretty much always 3
            3,
            time,
            config.vision_config.image_size,
            config.vision_config.image_size,
        )
        if num_videos is not None
        else None,
        video_input_mask=torch.tensor(
            [1] * (num_videos // batch * config.num_query_tokens)
            + [0] * (seq_len - num_videos // batch * config.num_query_tokens)
        )
        .unsqueeze(0)
        .expand(batch, -1)
        if num_videos is not None
        else None,
        labels=torch.ones(batch, seq_len).long(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    assert outputs.loss.size() == tuple()
    assert outputs.logits.size() == (batch, seq_len, config.text_config.vocab_size)


@pytest.mark.parametrize(
    # we use kwargs instead of generation_config so that we can use the model defaults
    "generate_kwargs",
    [
        {"num_beams": 1, "do_sample": False},  # greedy decoding
        {"penalty_alpha": 0.1, "top_k": 2},  # constrastive search
        {"num_beams": 1, "do_sample": True},  # multinomial sampling
        {"num_beams": 2, "do_sample": False},  # beam-search decoding
        {"num_beams": 2, "do_sample": True},  # beam-search multinomial sampling
        {
            "num_beams": 2,
            "num_beam_groups": 2,
            "diversity_penalty": 0.1,
        },  # diverse beam-search decoding
    ],
)
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [None, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "config",
    [
        Blip2Config(
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "projection_dim": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 8,
            },
            qformer_config={
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            text_config={
                "model_type": "opt",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "ffn_dim": 16,
                "num_attention_heads": 2,
            },
            num_query_tokens=4,
        ),
        Blip2Config(
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "projection_dim": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 8,
            },
            qformer_config={
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            text_config={
                "model_type": "t5",
                "d_model": 8,
                "d_kv": 4,
                "d_ff": 16,
                "num_layers": 2,
                "num_heads": 2,
                "decoder_start_token_id": 0,
            },
            num_query_tokens=4,
        ),
    ],
)
def test_v2_video_blip_for_cond_gen_generate(
    config, batch, num_videos, time, seq_len, generate_kwargs
):
    model = VideoBlipForConditionalGeneration(config)
    max_length = 5
    generated_ids = model.generate(
        input_ids=torch.ones(batch, seq_len).long(),
        attention_mask=torch.ones(batch, seq_len).long(),
        pixel_values=torch.rand(
            num_videos,
            # channel is pretty much always 3
            3,
            time,
            config.vision_config.image_size,
            config.vision_config.image_size,
        )
        if num_videos is not None
        else None,
        video_input_mask=torch.tensor(
            [1] * (num_videos // batch * config.num_query_tokens)
            + [0] * (seq_len - num_videos // batch * config.num_query_tokens)
        )
        .unsqueeze(0)
        .expand(batch, -1)
        if num_videos is not None
        else None,
        max_length=max_length,
        **generate_kwargs
    )
    assert generated_ids.size() == (batch, max_length)


@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("class_seq_len", [3, 5])
@pytest.mark.parametrize("class_batch_size", [None, 3])
@pytest.mark.parametrize("num_classes", [4, 6])
@pytest.mark.parametrize("prompt_seq_len", [16, 32])
@pytest.mark.parametrize("num_videos", [None, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "config",
    [
        Blip2Config(
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "projection_dim": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 8,
            },
            qformer_config={
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            text_config={
                "model_type": "opt",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "ffn_dim": 16,
                "num_attention_heads": 2,
            },
            num_query_tokens=4,
        ),
    ],
)
def test_v2_video_blip_for_cond_gen_classify(
    config,
    batch,
    num_videos,
    prompt_seq_len,
    num_classes,
    class_batch_size,
    class_seq_len,
    time,
):
    model = VideoBlipForConditionalGeneration(config).eval()
    classify_kwargs = {
        "prompt_input_ids": torch.ones(batch, prompt_seq_len).long(),
        "class_input_ids": torch.ones(num_classes, class_seq_len).long(),
        "prompt_attention_mask": torch.ones(batch, prompt_seq_len).long(),
        "pixel_values": torch.rand(
            num_videos,
            # channel is pretty much always 3
            3,
            time,
            config.vision_config.image_size,
            config.vision_config.image_size,
        )
        if num_videos is not None
        else None,
        "prompt_video_input_mask": torch.tensor(
            [1] * (num_videos // batch * config.num_query_tokens)
            + [0] * (prompt_seq_len - num_videos // batch * config.num_query_tokens)
        )
        .unsqueeze(0)
        .expand(batch, -1)
        if num_videos is not None
        else None,
        "class_attention_mask": torch.ones(num_classes, class_seq_len).long(),
    }
    log_likelihood = model.classify(**classify_kwargs)
    assert log_likelihood.size() == (batch, num_classes)
    class_batch_log_likelihood = model.classify(
        **classify_kwargs, class_batch_size=class_batch_size
    )
    assert log_likelihood.allclose(class_batch_log_likelihood)
