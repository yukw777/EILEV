import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    T5ForConditionalGeneration,
    logging,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    Seq2SeqLMOutput,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)
from transformers.models.opt.modeling_opt import (
    OPTDecoder,
    _expand_mask,
    _make_causal_mask,
)

logger = logging.get_logger(__name__)


def _make_video_causal_mask(
    video_causal_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allow text tokens to attend only to the last preceding video.

    :param video_causal_mask: a tensor of shape (batch, text_seq_len, video_seq_len)
        where text_seq_len + video_seq_len = seq_len
    :param attention_mask: optional attention mask of shape (batch, seq_len).
        useful for handling padding
    :param dtype: dtype of the resulting mask

    :returns: additive video causal mask of shape (batch, 1, tgt_seq_len, src_seq_len)
    """
    inverted_mask = 1.0 - video_causal_mask.to(dtype=dtype)

    # (batch, text_seq_len, video_seq_len)
    mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )
    _, text_seq_len, video_seq_len = video_causal_mask.size()
    # (batch, seq_len, seq_len)
    mask = F.pad(mask, (0, text_seq_len, video_seq_len, 0))
    if attention_mask is not None:
        mask *= attention_mask.unsqueeze(-1)

    # (batch, seq_len, seq_len)
    return mask.unsqueeze(1)


class VideoOPTDecoder(OPTDecoder):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        video_causal_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        """Mostly copied from OPTDecoder.forward().

        Look at the official doc for OPTDecoder for more details on undocumented
        parameters.

        :param video_causal_mask: a tensor of shape (batch, text_seq_len, video_seq_len)
            where text_seq_len + video_seq_len = seq_len
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds "
                "at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # type: ignore
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            video_causal_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. "
                    "Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} "
                        f"layers, but it is for {head_mask.size()[0]}."  # type: ignore
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (  # type: ignore
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # type: ignore

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        video_causal_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if video_causal_mask is not None:
            expanded_video_causal_mask = _make_video_causal_mask(
                video_causal_mask, attention_mask, inputs_embeds.dtype
            )
            combined_attention_mask += expanded_video_causal_mask

        return combined_attention_mask


class VideoOPTModel(OPTModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = VideoOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class VideoOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = VideoOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        video_causal_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        """Mostly copied from OPTForCausalLM.forward().

        Look at the official doc for OPTForCausalLM for more details on undocumented
        parameters.

        :param video_causal_mask: a tensor of shape (batch, text_seq_len, video_seq_len)
            where text_seq_len + video_seq_len = seq_len
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video_causal_mask=video_causal_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        video_causal_mask=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )
        # if `video_causal_mask` is passed, we only want to use them in
        # the 1st generation step, after which we should proceed as if
        # we're performing regular text generation w/out video causal masks.
        if video_causal_mask is not None and past_key_values is None:
            inputs["video_causal_mask"] = video_causal_mask
        return inputs


class VideoBlipVisionModel(Blip2VisionModel):
    """A simple, augmented version of Blip2VisionModel to handle multiple
    videos."""

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        """Flatten `pixel_values` along the batch and time dimension, pass it
        through the original vision model, then unflatten it back.

        :param pixel_values: a tensor of shape
            (batch, num_videos, channel, time, height, width)

        :returns:
            last_hidden_state: a tensor of shape
                (batch, num_videos, time * seq_len, hidden_size)
            pooler_output: a tensor of shape (batch, num_videos, time, hidden_size)
            hidden_states:
                a tuple of tensors of shape
                (batch, num_videos, time * seq_len, hidden_size),
                one for the output of the embeddings + one for each layer
            attentions:
                a tuple of tensors of shape
                (batch, num_videos, time, num_heads, seq_len, seq_len),
                one for each layer
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch, num_videos, _, time, _, _ = pixel_values.size()

        # flatten along the batch and time dimension to create a tensor of shape
        # (batch * num_videos * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 1, 3, 2, 4, 5).flatten(end_dim=2)

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # now restore the original dimensions
        # vision_outputs.last_hidden_state is of shape
        # (batch * num_videos * time, seq_len, hidden_size)
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(
            batch, num_videos, time * seq_len, -1
        )
        # vision_outputs.pooler_output is of shape
        # (batch * num_videos * time, hidden_size)
        pooler_output = vision_outputs.pooler_output.view(batch, num_videos, time, -1)
        # hidden_states is a tuple of tensors of shape
        # (batch * num_videos * time, seq_len, hidden_size)
        hidden_states = (
            tuple(
                hidden.view(batch, num_videos, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )
        # attentions is a tuple of tensors of shape
        # (batch * num_videos * time, num_heads, seq_len, seq_len)
        attentions = (
            tuple(
                hidden.view(batch, num_videos, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions
            )
            if vision_outputs.attentions is not None
            else None
        )
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return (last_hidden_state, pooler_output, hidden_states, attentions)


def _make_video_causal_encoder_attn_mask(
    video_causal_mask: torch.LongTensor | None,
    encoder_attention_mask: torch.LongTensor | None,
) -> torch.Tensor | None:
    """Allow decoder to attend only to the last preceding video.

    :param video_causal_mask: a tensor of shape (batch, text_seq_len, video_seq_len)
        where text_seq_len + video_seq_len = src_seq_len
    :param encoder_attention_mask: optional encoder attention mask of shape
        (batch, seq_len) for padding

    :returns: video causal encoder attention mask of shape (batch, src_seq_len)
    """
    if video_causal_mask is None:
        return encoder_attention_mask
    batch, text_seq_len, video_seq_len = video_causal_mask.size()
    seq_len = text_seq_len + video_seq_len
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(  # type: ignore
            batch,
            text_seq_len + video_seq_len,
            dtype=video_causal_mask.dtype,
            device=video_causal_mask.device,
        )
    assert encoder_attention_mask is not None
    assert seq_len == encoder_attention_mask.size(1)

    # we want the decoder to cross-attend only to the videos that the last non-padding
    # text token attended to.
    # first, figure out the indices of the last non-padding text token, which is the
    # index of the first 0 in encoder_attention_mask. Note that argmin() returns the
    # first minimal value if there are multiple options, which is what we want.
    last_non_pad_text_token_indices = (
        encoder_attention_mask[:, video_seq_len:].argmin(dim=1) - 1
    )
    # if there is no padding, argmin() above returns 0, but we want to select the last
    # index.
    last_non_pad_text_token_indices[encoder_attention_mask.sum(dim=1) == seq_len] = (
        text_seq_len - 1
    )
    # second, select the video causal masks for the last non-padding text tokens
    # (batch, video_seq_len)
    video_causal_encoder_attention_mask = video_causal_mask[
        torch.arange(batch), last_non_pad_text_token_indices
    ]
    # third, pad it with 1
    # (batch, seq_len)
    video_causal_encoder_attention_mask = F.pad(
        video_causal_encoder_attention_mask, (0, text_seq_len), value=1
    )
    # lastly, multiply it by encoder_attention_mask to account for padding text tokens
    video_causal_encoder_attention_mask *= encoder_attention_mask
    return video_causal_encoder_attention_mask


class VideoT5ForConditionalGeneration(T5ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        video_causal_mask: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        decoder_head_mask: torch.FloatTensor | None = None,
        cross_attn_head_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput:
        """Mostly copied from T5ForConditionalGeneration.forward()

        Look at the official doc T5ForConditionalGeneration for for more details on
        undocumented parameters.

        :param video_causal_mask: a tensor of shape (batch, text_seq_len, video_seq_len)
            where text_seq_len + video_seq_len = seq_len
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args -
        # head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(
                    "head_mask was separated into two input args - head_mask, "
                    "decoder_head_mask",
                    FutureWarning,
                )
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,  # type: ignore # noqa: E501
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,  # type: ignore # noqa: E501
            )

        hidden_states = encoder_outputs[0]  # type: ignore

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)  # type: ignore
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)  # type: ignore # noqa: E501
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)  # type: ignore # noqa: E501
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(  # type: ignore
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=_make_video_causal_encoder_attn_mask(
                video_causal_mask, attention_mask
            ),
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)  # type: ignore
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586 # noqa: E501
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)  # type: ignore
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666 # noqa: E501

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,  # type: ignore
            encoder_hidden_states=encoder_outputs.hidden_states,  # type: ignore
            encoder_attentions=encoder_outputs.attentions,  # type: ignore
        )


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.vision_model = VideoBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = VideoOPTForCausalLM(config.text_config)
        else:
            language_model = VideoT5ForConditionalGeneration(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        video_causal_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | Blip2ForConditionalGenerationModelOutput:
        """Mostly copied from Blip2ForConditionalGeneration.forward()

        Look at the official doc for Blip2ForConditionalGeneration for more details on
        undocumented parameters.

        :param pixel_values: a tensor of shape
            (batch, num_videos, channel, time, height, width)
        :param video_causal_mask: a tensor of shape (batch, seq_len, num_videos)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape
        # (batch_size, num_videos, time * vision_seq_len, vision_hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer,
        # using the image embeddings for cross-attention
        # (batch_size * num_videos, time * vision_seq_len, vision_hidden_size)
        image_embeds = image_embeds.flatten(end_dim=1)
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # (batch_size * num_videos, num_query_tokens, qformer_hidden_size)
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and
        # the prompt
        batch_size, num_videos, _, _, _, _ = pixel_values.size()
        language_model_inputs = self.language_projection(
            query_output.view(batch_size, num_videos * self.config.num_query_tokens, -1)
        )
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            device=language_model_inputs.device,
            dtype=torch.long,
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # (batch_size, num_videos * num_query_tokens + num_tokens, qformer_hidden_size)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # type: ignore
        attention_mask = torch.cat(
            [
                language_model_attention_mask,
                attention_mask.to(language_model_inputs.device),  # type: ignore
            ],
            dim=1,
        )
        if video_causal_mask is None:
            video_causal_mask = torch.ones(
                batch_size, input_ids.size(1), num_videos, dtype=torch.long
            )  # type: ignore
        video_causal_mask = video_causal_mask.to(  # type: ignore
            language_model_inputs.device
        ).repeat_interleave(self.config.num_query_tokens, dim=2)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                video_causal_mask=video_causal_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account
            # the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)  # type: ignore
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size),
                    shift_labels.view(-1),
                )
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                video_causal_mask=video_causal_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        video_causal_mask: torch.LongTensor | None = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Mostly copied from Blip2ForConditionalGeneration.generate()

        Look at the official doc for Blip2ForConditionalGeneration for more details on
        undocumented parameters.

        :param video_causal_mask: a tensor of shape (batch, seq_len, num_videos)
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape
        # (batch_size, num_videos, time * vision_seq_len, vision_hidden_size)
        image_embeds = self.vision_model(
            pixel_values, return_dict=True
        ).last_hidden_state

        # step 2: forward the query tokens through the QFormer,
        # using the image embeddings for cross-attention
        # (batch_size * num_videos, time * vision_seq_len, vision_hidden_size)
        image_embeds = image_embeds.flatten(end_dim=1)
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        # (batch_size * num_videos, num_query_tokens, qformer_hidden_size)
        query_output = query_outputs.last_hidden_state

        # step 3: use the language model, conditioned on the query outputs and
        # the prompt
        batch_size, num_videos, _, _, _, _ = pixel_values.size()
        language_model_inputs = self.language_projection(
            query_output.view(batch_size, num_videos * self.config.num_query_tokens, -1)
        )
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)  # type: ignore
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # type: ignore
        attention_mask = torch.cat(
            [
                language_attention_mask,
                attention_mask.to(language_attention_mask.device),  # type: ignore
            ],
            dim=1,
        )
        if video_causal_mask is None:
            video_causal_mask = torch.ones(
                batch_size, input_ids.size(1), num_videos  # type: ignore
            )
        video_causal_mask = video_causal_mask.to(  # type: ignore
            language_model_inputs.device
        ).repeat_interleave(self.config.num_query_tokens, dim=2)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            video_causal_mask=video_causal_mask,
            **generate_kwargs,
        )

        return outputs
