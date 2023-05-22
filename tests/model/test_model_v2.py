import pytest
import torch
from transformers import Blip2Config, Blip2VisionConfig, OPTConfig, T5Config

from video_blip.model.v2 import (
    VideoBlipForConditionalGeneration,
    VideoBlipVisionModel,
    VideoOPTDecoder,
    VideoOPTForCausalLM,
    VideoT5ForConditionalGeneration,
    _make_video_causal_encoder_attn_mask,
    _make_video_causal_mask,
)


@pytest.mark.parametrize(
    "video_causal_mask,attention_mask,dtype,expected",
    [
        (
            torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]]),
            torch.ones(1, 7).long(),
            torch.float,
            torch.tensor(
                [
                    [0.0] * 7,
                    [0.0] * 7,
                    [0.0] * 7,
                    [0.0] * 7,
                    [0.0] * 2
                    + [torch.finfo(torch.float).min, torch.finfo(torch.float).min]
                    + [0.0] * 3,
                    [0.0] * 2
                    + [torch.finfo(torch.float).min, torch.finfo(torch.float).min]
                    + [0.0] * 3,
                    [torch.finfo(torch.float).min, torch.finfo(torch.float).min]
                    + [0.0] * 5,
                ]
            ).view(1, 1, 7, 7),
        ),
        (
            torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]]),
            torch.ones(1, 7).long(),
            torch.float16,
            torch.tensor(
                [
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 2
                    + [torch.finfo(torch.float16).min, torch.finfo(torch.float16).min]
                    + [0] * 3,
                    [0] * 2
                    + [torch.finfo(torch.float16).min, torch.finfo(torch.float16).min]
                    + [0] * 3,
                    [torch.finfo(torch.float16).min, torch.finfo(torch.float16).min]
                    + [0] * 5,
                ]
            ).view(1, 1, 7, 7),
        ),
        (
            torch.tensor(
                [
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            torch.tensor([[1] * 9 + [0] * 2, [1] * 11, [1] * 8 + [0] * 3]),
            torch.float,
            torch.tensor(
                [
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float).min] * 4 + [0.0] * 5,
                        [0.0] * 2 + [torch.finfo(torch.float).min] * 4 + [0.0] * 5,
                        [torch.finfo(torch.float).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float).min] * 2
                        + [0.0] * 5,
                        [0.0] * 11,
                        [0.0] * 11,
                    ],
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float).min] * 4 + [0.0] * 5,
                        [torch.finfo(torch.float).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float).min] * 2
                        + [0.0] * 5,
                        [torch.finfo(torch.float).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float).min] * 2
                        + [0.0] * 5,
                        [torch.finfo(torch.float).min] * 4 + [0.0] * 7,
                        [torch.finfo(torch.float).min] * 4 + [0.0] * 7,
                    ],
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float).min] * 4 + [0.0] * 5,
                        [0.0] * 2 + [torch.finfo(torch.float).min] * 4 + [0.0] * 5,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                    ],
                ]
            ).unsqueeze(1),
        ),
        (
            torch.tensor(
                [
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            torch.tensor([[1] * 9 + [0] * 2, [1] * 11, [1] * 8 + [0] * 3]),
            torch.float16,
            torch.tensor(
                [
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float16).min] * 4 + [0.0] * 5,
                        [0.0] * 2 + [torch.finfo(torch.float16).min] * 4 + [0.0] * 5,
                        [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 5,
                        [0.0] * 11,
                        [0.0] * 11,
                    ],
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float16).min] * 4 + [0.0] * 5,
                        [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 5,
                        [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 2
                        + [torch.finfo(torch.float16).min] * 2
                        + [0.0] * 5,
                        [torch.finfo(torch.float16).min] * 4 + [0.0] * 7,
                        [torch.finfo(torch.float16).min] * 4 + [0.0] * 7,
                    ],
                    [
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 2 + [torch.finfo(torch.float16).min] * 4 + [0.0] * 5,
                        [0.0] * 2 + [torch.finfo(torch.float16).min] * 4 + [0.0] * 5,
                        [0.0] * 11,
                        [0.0] * 11,
                        [0.0] * 11,
                    ],
                ]
            ).unsqueeze(1),
        ),
    ],
)
def test_make_video_causal_mask(video_causal_mask, attention_mask, dtype, expected):
    assert _make_video_causal_mask(video_causal_mask, attention_mask, dtype).equal(
        expected
    )


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("video_seq_len", [1, 8])
@pytest.mark.parametrize("text_seq_len", [1, 5])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize(
    "config",
    [
        OPTConfig(
            hidden_size=8, num_hidden_layers=2, ffn_dim=16, num_attention_heads=2
        ),
        OPTConfig(
            hidden_size=16, num_hidden_layers=4, ffn_dim=32, num_attention_heads=4
        ),
    ],
)
def test_v2_video_opt_decoder_forward(
    config: OPTConfig,
    batch: int,
    text_seq_len: int,
    video_seq_len: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoOPTDecoder(config)
    outputs = model(
        input_ids=torch.ones(batch, video_seq_len + text_seq_len).long(),
        attention_mask=torch.ones(batch, video_seq_len + text_seq_len).long(),
        video_causal_mask=torch.ones(batch, text_seq_len, video_seq_len).long(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    seq_len = text_seq_len + video_seq_len
    assert outputs.last_hidden_state.size() == (batch, seq_len, config.hidden_size)
    assert len(outputs.past_key_values) == config.num_hidden_layers
    for t1, t2 in outputs.past_key_values:
        assert t1.size() == (
            batch,
            config.num_attention_heads,
            seq_len,
            config.hidden_size // config.num_attention_heads,
        )
        assert t2.size() == (
            batch,
            config.num_attention_heads,
            seq_len,
            config.hidden_size // config.num_attention_heads,
        )

    if output_attentions:
        assert len(outputs.attentions) == config.num_hidden_layers
        for attn in outputs.attentions:
            assert attn.size() == (batch, config.num_attention_heads, seq_len, seq_len)
    else:
        assert outputs.attentions is None

    if output_hidden_states:
        # num_hidden_layers + 1 for embeddings
        assert len(outputs.hidden_states) == config.num_hidden_layers + 1
        for hidden in outputs.hidden_states:
            assert hidden.size() == (batch, seq_len, config.hidden_size)
    else:
        assert outputs.hidden_states is None


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("video_seq_len", [1, 8])
@pytest.mark.parametrize("text_seq_len", [1, 5])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize(
    "config",
    [
        OPTConfig(
            hidden_size=8, num_hidden_layers=2, ffn_dim=16, num_attention_heads=2
        ),
        OPTConfig(
            hidden_size=16, num_hidden_layers=4, ffn_dim=32, num_attention_heads=4
        ),
    ],
)
def test_v2_video_opt_for_causal_lm_forward(
    config: OPTConfig,
    batch: int,
    text_seq_len: int,
    video_seq_len: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoOPTForCausalLM(config)
    outputs = model(
        input_ids=torch.ones(batch, video_seq_len + text_seq_len).long(),
        attention_mask=torch.ones(batch, video_seq_len + text_seq_len).long(),
        video_causal_mask=torch.ones(batch, text_seq_len, video_seq_len).long(),
        labels=torch.ones(batch, video_seq_len + text_seq_len).long(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    assert outputs.loss.size() == tuple()
    seq_len = text_seq_len + video_seq_len
    assert outputs.logits.size() == (batch, seq_len, config.vocab_size)
    assert len(outputs.past_key_values) == config.num_hidden_layers
    for t1, t2 in outputs.past_key_values:
        assert t1.size() == (
            batch,
            config.num_attention_heads,
            seq_len,
            config.hidden_size // config.num_attention_heads,
        )
        assert t2.size() == (
            batch,
            config.num_attention_heads,
            seq_len,
            config.hidden_size // config.num_attention_heads,
        )

    if output_attentions:
        assert len(outputs.attentions) == config.num_hidden_layers
        for attn in outputs.attentions:
            assert attn.size() == (batch, config.num_attention_heads, seq_len, seq_len)
    else:
        assert outputs.attentions is None

    if output_hidden_states:
        # num_hidden_layers + 1 for embeddings
        assert len(outputs.hidden_states) == config.num_hidden_layers + 1
        for hidden in outputs.hidden_states:
            assert hidden.size() == (batch, seq_len, config.hidden_size)
    else:
        assert outputs.hidden_states is None


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [1, 5])
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
def test_v2_video_blip_vision_model_forward(
    config: Blip2VisionConfig,
    batch: int,
    num_videos: int,
    time: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoBlipVisionModel(config)
    outputs = model(
        pixel_values=torch.rand(
            # channel is pretty much always 3
            batch,
            num_videos,
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
        batch,
        num_videos,
        time * num_tokens,
        config.hidden_size,
    )
    assert pooler_output.size() == (batch, num_videos, time, config.hidden_size)

    if output_attentions:
        assert len(attentions) == config.num_hidden_layers
        for attn in attentions:
            assert attn.size() == (
                batch,
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
            assert hidden.size() == (
                batch,
                num_videos,
                time * num_tokens,
                config.hidden_size,
            )
    else:
        assert hidden_states is None


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("seq_len", [1, 16])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [1, 5])
@pytest.mark.parametrize("batch", [1, 4])
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
    num_videos: int,
    time: int,
    seq_len: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoBlipForConditionalGeneration(config)
    outputs = model(
        torch.rand(
            # channel is pretty much always 3
            batch,
            num_videos,
            3,
            time,
            config.vision_config.image_size,
            config.vision_config.image_size,
        ),
        torch.ones(batch, seq_len).long(),
        attention_mask=torch.ones(batch, seq_len).long(),
        video_causal_mask=torch.ones(batch, seq_len, num_videos).long(),
        labels=torch.ones(batch, seq_len).long(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    assert outputs.loss.size() == tuple()
    assert outputs.logits.size() == (batch, seq_len, config.text_config.vocab_size)
    assert outputs.qformer_outputs.last_hidden_state.size() == (
        batch * num_videos,
        config.num_query_tokens,
        config.qformer_config.hidden_size,
    )
    assert outputs.qformer_outputs.pooler_output.size() == (
        batch * num_videos,
        config.qformer_config.hidden_size,
    )
    if output_attentions:
        assert (
            len(outputs.qformer_outputs.attentions)
            == config.qformer_config.num_hidden_layers
        )
        for attn in outputs.qformer_outputs.attentions:
            assert attn.size() == (
                batch * num_videos,
                config.qformer_config.num_attention_heads,
                config.num_query_tokens,
                config.num_query_tokens,
            )
    else:
        assert outputs.qformer_outputs.attentions is None

    if output_attentions:
        assert (
            len(outputs.qformer_outputs.cross_attentions)
            == config.qformer_config.num_hidden_layers
            // config.qformer_config.cross_attention_frequency
        )
        num_vision_tokens = (
            (config.vision_config.image_size // config.vision_config.patch_size) ** 2
            + 1
        ) * time
        for cross_attn in outputs.qformer_outputs.cross_attentions:
            assert cross_attn.size() == (
                batch * num_videos,
                config.qformer_config.num_attention_heads,
                config.num_query_tokens,
                num_vision_tokens,
            )
    else:
        assert outputs.qformer_outputs.cross_attentions is None


@pytest.mark.parametrize(
    # we use kwargs instead of generation_config so that we can use the model defaults
    "generate_kwargs",
    [
        {"num_beams": 1, "do_sample": False},  # greedy decoding
        {"penalty_alpha": 0.1, "top_k": 2},  # constrastive search
        {"num_beams": 1, "do_sample": True},  # multinomial sampling
        {"num_beams": 2, "do_sample": False},  # beam-search decoding
        {"num_beams": 2, "do_sample": True},  # beam-search multinomial sampling
        {"num_beams": 2, "num_beam_groups": 2},  # diverse beam-search decoding
    ],
)
@pytest.mark.parametrize("seq_len", [1, 16])
@pytest.mark.parametrize("time", [1, 8])
@pytest.mark.parametrize("num_videos", [1, 5])
@pytest.mark.parametrize("batch", [1, 4])
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
        torch.rand(
            # channel is pretty much always 3
            batch,
            num_videos,
            3,
            time,
            config.vision_config.image_size,
            config.vision_config.image_size,
        ),
        input_ids=torch.ones(batch, seq_len).long(),
        attention_mask=torch.ones(batch, seq_len).long(),
        video_causal_mask=torch.ones(batch, seq_len, num_videos).long(),
        max_length=max_length,
        **generate_kwargs
    )
    assert generated_ids.size() == (batch, max_length)


@pytest.mark.parametrize(
    "video_causal_mask,encoder_attention_mask,expected",
    [
        (None, None, None),
        (None, torch.ones(2, 8).long(), torch.ones(2, 8).long()),
        (
            torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]]),
            torch.ones(1, 7).long(),
            torch.tensor([[0, 0, 1, 1, 1, 1, 1]]),
        ),
        (
            torch.tensor(
                [
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            torch.tensor([[1] * 9 + [0] * 2, [1] * 11, [1] * 8 + [0] * 3]),
            torch.tensor(
                [
                    [0, 0, 1, 1, 0, 0] + [1] * 3 + [0] * 2,
                    [0, 0, 0, 0, 1, 1] + [1] * 5,
                    [1, 1, 0, 0, 0, 0] + [1] * 2 + [0] * 3,
                ]
            ),
        ),
    ],
)
def test_make_video_causal_encoder_attn_mask(
    video_causal_mask, encoder_attention_mask, expected
):
    mask = _make_video_causal_encoder_attn_mask(
        video_causal_mask, encoder_attention_mask
    )
    if expected is None:
        assert mask is None
    else:
        assert mask.equal(expected)


@pytest.mark.parametrize("output_hidden_states", [True, False])
@pytest.mark.parametrize("output_attentions", [True, False])
@pytest.mark.parametrize("video_seq_len", [1, 8])
@pytest.mark.parametrize("text_seq_len", [1, 5])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize(
    "config",
    [
        T5Config(
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_layers=2,
            num_heads=2,
            decoder_start_token_id=0,
        ),
        T5Config(
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=4,
            num_heads=4,
            decoder_start_token_id=0,
        ),
    ],
)
def test_v2_video_t5_for_cond_gen_forward(
    config: T5Config,
    batch: int,
    text_seq_len: int,
    video_seq_len: int,
    output_attentions: bool,
    output_hidden_states: bool,
) -> None:
    model = VideoT5ForConditionalGeneration(config)
    outputs = model(
        input_ids=torch.ones(batch, video_seq_len + text_seq_len).long(),
        attention_mask=torch.ones(batch, video_seq_len + text_seq_len).long(),
        video_causal_mask=torch.ones(batch, text_seq_len, video_seq_len).long(),
        labels=torch.ones(batch, video_seq_len + text_seq_len).long(),
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    assert outputs.loss.size() == tuple()
    seq_len = text_seq_len + video_seq_len
    assert outputs.logits.size() == (batch, seq_len, config.vocab_size)
    assert len(outputs.past_key_values) == config.num_hidden_layers
    for tensors in outputs.past_key_values:
        assert len(tensors) == 4
        for tensor in tensors:
            assert tensor.size() == (batch, config.num_heads, seq_len, config.d_kv)

    if output_attentions:
        assert len(outputs.decoder_attentions) == config.num_layers
        for attn in outputs.decoder_attentions:
            assert attn.size() == (batch, config.num_heads, seq_len, seq_len)
        assert len(outputs.cross_attentions) == config.num_layers
        for attn in outputs.cross_attentions:
            assert attn.size() == (batch, config.num_heads, seq_len, seq_len)
        assert len(outputs.encoder_attentions) == config.num_layers
        for attn in outputs.encoder_attentions:
            assert attn.size() == (batch, config.num_heads, seq_len, seq_len)
    else:
        assert outputs.decoder_attentions is None
        assert outputs.cross_attentions is None
        assert outputs.encoder_attentions is None

    if output_hidden_states:
        # num_hidden_layers + 1 for embeddings
        assert len(outputs.decoder_hidden_states) == config.num_hidden_layers + 1
        for hidden in outputs.decoder_hidden_states:
            assert hidden.size() == (batch, seq_len, config.d_model)
        # num_hidden_layers + 1 for embeddings
        assert len(outputs.encoder_hidden_states) == config.num_hidden_layers + 1
        for hidden in outputs.encoder_hidden_states:
            assert hidden.size() == (batch, seq_len, config.d_model)
    else:
        assert outputs.decoder_hidden_states is None
        assert outputs.encoder_hidden_states is None
