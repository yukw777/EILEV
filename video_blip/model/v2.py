import random

import torch
import torch.nn as nn
from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    logging,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)
from transformers.models.opt.modeling_opt import OPTDecoder

logger = logging.get_logger(__name__)


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
            batch_size,
            seq_length,
            past_key_values_length,
            inputs_embeds.device,
            inputs_embeds.dtype,
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
        attention_mask: torch.Tensor | None,
        video_causal_mask: torch.Tensor | None,
        batch: int,
        tgt_seq_len: int,
        past_key_values_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create an additive decoder attention mask that is a combination of
        causal, video causal and attention masks.

        :param attention_mask: optional attention mask of shape (batch, src_seq_len).
            useful for handling padding
        :param video_causal_mask: optional video causal mask of shape
            (batch, text_seq_len, video_seq_len)
        :param batch: batch dimension size
        :param tgt_seq_len: target sequence length
        :param past_key_values_length: length of cached past key values,
            src_seq_len = past_key_values_length + tgt_seq_len
        :param device: device of the resulting mask
        :param dtype: dtype of the resulting mask

        :returns: additive attention mask of shape (batch, 1, tgt_seq_len, src_seq_len)
        """
        src_seq_len = past_key_values_length + tgt_seq_len
        combined_attention_mask = torch.ones(
            batch, tgt_seq_len, src_seq_len, device=device, dtype=dtype
        )
        if tgt_seq_len > 1:
            if video_causal_mask is not None:
                num_videos = (
                    video_causal_mask.size(2) // self.config.qformer_num_query_tokens
                )
                # (video_seq_len, video_seq_len)
                causal_mask = (
                    torch.eye(num_videos, device=device)
                    .repeat_interleave(self.config.qformer_num_query_tokens, dim=0)
                    .repeat_interleave(self.config.qformer_num_query_tokens, dim=1)
                )
                video_seq_len = causal_mask.size(0)
                # (tgt_seq_len, video_seq_len)
                causal_mask = torch.cat(
                    [
                        causal_mask,
                        torch.ones(
                            tgt_seq_len - video_seq_len,
                            video_seq_len,
                            device=causal_mask.device,
                        ),
                    ],
                    dim=0,
                )
                # (tgt_seq_len, tgt_seq_len)
                causal_mask = torch.cat(
                    [
                        causal_mask,
                        # (tgt_seq_len, tgt_seq_len - video_seq_len)
                        torch.cat(
                            [
                                torch.zeros(
                                    video_seq_len,
                                    tgt_seq_len - video_seq_len,
                                    device=causal_mask.device,
                                ),
                                torch.ones(
                                    tgt_seq_len - video_seq_len,
                                    tgt_seq_len - video_seq_len,
                                    device=video_causal_mask.device,
                                ).tril(),
                            ],
                            dim=0,
                        ),
                    ],
                    dim=1,
                )
            else:
                # (tgt_seq_len, tgt_seq_len)
                causal_mask = torch.ones(tgt_seq_len, tgt_seq_len, device=device).tril()
            # (tgt_seq_len, src_seq_len)
            causal_mask = torch.cat(
                [
                    torch.ones(tgt_seq_len, past_key_values_length, device=device),
                    causal_mask,
                ],
                dim=-1,
            )
            # (batch, tgt_seq_len, src_seq_len)
            causal_mask = causal_mask[None, :, :].expand(
                batch, tgt_seq_len, src_seq_len
            )

            combined_attention_mask *= causal_mask

        if video_causal_mask is not None:
            _, _, video_seq_len = video_causal_mask.size()
            # (batch, tgt_seq_len, video_seq_len)
            expanded_video_causal_mask = torch.cat(
                [
                    torch.ones(
                        batch,
                        video_seq_len,
                        video_seq_len,
                        device=video_causal_mask.device,
                    ),
                    video_causal_mask,
                ],
                dim=1,
            )[:, -tgt_seq_len:]
            # (batch, tgt_seq_len, src_seq_len)
            expanded_video_causal_mask = torch.cat(
                [
                    expanded_video_causal_mask,
                    torch.ones(
                        batch,
                        tgt_seq_len,
                        src_seq_len - video_seq_len,
                        device=expanded_video_causal_mask.device,
                    ),
                ],
                dim=-1,
            )
            combined_attention_mask *= expanded_video_causal_mask

        if attention_mask is not None:
            # (batch, tgt_seq_len, src_seq_len)
            expanded_attn_mask = attention_mask[:, None, :].expand(
                batch, tgt_seq_len, -1
            )
            combined_attention_mask *= expanded_attn_mask

        inverted_mask = 1.0 - combined_attention_mask

        # (batch, 1, tgt_seq_len, src_seq_len)
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        ).unsqueeze(1)


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


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config) -> None:
        # This version is trained only on autoregressive language modeling,
        # so only decoder only LMs are supported
        assert config.use_decoder_only_language_model
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
        config.text_config.qformer_num_query_tokens = config.num_query_tokens
        language_model = VideoOPTForCausalLM(config.text_config)
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
