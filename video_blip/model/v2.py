import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)


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
            (num_videos, channel, time, height, width)

        :returns:
            last_hidden_state: a tensor of shape
                (num_videos, time * seq_len, hidden_size)
            pooler_output: a tensor of shape (num_videos, time, hidden_size)
            hidden_states:
                a tuple of tensors of shape
                (num_videos, time * seq_len, hidden_size),
                one for the output of the embeddings + one for each layer
            attentions:
                a tuple of tensors of shape
                (num_videos, time, num_heads, seq_len, seq_len),
                one for each layer
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        num_videos, _, time, _, _ = pixel_values.size()

        # flatten along the video and time dimension to create a tensor of shape
        # (num_videos * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # now restore the original dimensions
        # vision_outputs.last_hidden_state is of shape
        # (num_videos * time, seq_len, hidden_size)
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(
            num_videos, time * seq_len, -1
        )
        # vision_outputs.pooler_output is of shape
        # (num_videos * time, hidden_size)
        pooler_output = vision_outputs.pooler_output.view(num_videos, time, -1)
        # hidden_states is a tuple of tensors of shape
        # (num_videos * time, seq_len, hidden_size)
        hidden_states = (
            tuple(
                hidden.view(num_videos, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )
        # attentions is a tuple of tensors of shape
        # (num_videos * time, num_heads, seq_len, seq_len)
        attentions = (
            tuple(
                hidden.view(num_videos, time, -1, seq_len, seq_len)
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
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | Blip2ForConditionalGenerationModelOutput:
        """Mostly copied from Blip2ForConditionalGeneration.forward()

        Look at the official doc for Blip2ForConditionalGeneration for more details on
        undocumented parameters.

        :param pixel_values: a tensor of shape
            (num_videos, channel, time, height, width)
        :param video_input_mask: a tensor of shape (batch, seq_len)
        """
        if pixel_values is not None:
            # if pixel_values is given, we need video_input_mask
            assert video_input_mask is not None
            video_input_mask = video_input_mask.bool()

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        vision_outputs: BaseModelOutputWithPooling | None = None
        video_features: torch.Tensor | None = None
        query_outputs: BaseModelOutputWithPoolingAndCrossAttentions | None = None
        if pixel_values is not None:
            # step 1: forward the images through the vision encoder
            # to get image embeddings of shape
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            assert vision_outputs is not None
            # (num_videos, time * vision_seq_len, vision_hidden_size)
            image_embeds = vision_outputs[0]

            # step 2: forward the query tokens through the QFormer,
            # using the image embeddings for cross-attention
            # (num_videos, time * vision_seq_len, vision_hidden_size)
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
            # (num_videos, num_query_tokens, qformer_hidden_size)
            query_output = query_outputs[0]

            # step 3: project the qformer tokens to the language model space
            num_videos = pixel_values.size(0)
            # (num_videos * num_query_tokens, text_hidden_size)
            video_features = self.language_projection(
                query_output.view(num_videos * self.config.num_query_tokens, -1)
            )
        # (batch_size, seq_len, text_hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if video_features is not None:
            # we need to clone inputs_embeds first since it may require gradients
            # and index assignment is an inplace operation
            tmp_inputs_embeds = inputs_embeds.clone()
            tmp_inputs_embeds[video_input_mask] = video_features.to(
                # for mixed-precision training
                dtype=tmp_inputs_embeds.dtype
            )
            inputs_embeds = tmp_inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # type: ignore

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
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
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Mostly copied from Blip2ForConditionalGeneration.generate()

        Look at the official doc for Blip2ForConditionalGeneration for more details on
        undocumented parameters.

        :param video_input_mask: a tensor of shape (batch, seq_len)
        """
        # at least one of input_ids or pixel_values should be given
        assert not (input_ids is None and pixel_values is None)
        if pixel_values is not None:
            # if pixel_values is given, we need video_input_mask
            assert video_input_mask is not None
            video_input_mask = video_input_mask.bool()
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        video_features: torch.Tensor | None = None
        if pixel_values is not None:
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape
            # (num_videos, time * vision_seq_len, vision_hidden_size)
            image_embeds = self.vision_model(
                pixel_values, return_dict=True
            ).last_hidden_state

            # step 2: forward the query tokens through the QFormer,
            # using the image embeddings for cross-attention
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
            # (num_videos, num_query_tokens, qformer_hidden_size)
            query_output = query_outputs.last_hidden_state

            # step 3: project the qformer tokens to the language model space
            num_videos = pixel_values.size(0)
            # (batch_size * num_videos * num_query_tokens, text_hidden_size)
            video_features = self.language_projection(
                query_output.view(num_videos * self.config.num_query_tokens, -1)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # type: ignore

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if video_features is not None:
            inputs_embeds[video_input_mask] = video_features

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    @torch.no_grad()
    def classify(
        self,
        prompt_input_ids: torch.Tensor,
        class_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        prompt_video_input_mask: torch.Tensor | None = None,
        class_attention_mask: torch.Tensor | None = None,
        class_batch_size: int | None = None,
    ) -> torch.Tensor:
        """

        :param prompt_input_ids: tensor of shape (batch, prompt_seq_len),
            padding should be left-sided.
        :param class_input_ids: tensor of shape (num_classes, class_seq_len)
        :param prompt_attention_mask: tensor of shape (batch, prompt_seq_len)
        :param pixel_values: tensor of shape
            (num_videos, channel, time, height, width)
        :param prompt_video_input_mask: tensor of shape (batch, prompt_seq_len)
        :param class_attention_mask: tensor of shape (num_classes, class_seq_len)
        :param class_batch_size: batch size for processing classes
        :return: log likelihoods for the classes
        """
        # only support decoder only language models for now
        assert self.config.use_decoder_only_language_model

        if pixel_values is not None:
            # if pixel_values is given, we need video_input_mask
            assert prompt_video_input_mask is not None
            prompt_video_input_mask = prompt_video_input_mask.bool()

        prompt_video_features: torch.Tensor | None = None
        if pixel_values is not None:
            # step 1: forward the images through the vision encoder
            # to get image embeddings of shape
            # (num_videos, time * vision_seq_len, vision_hidden_size)
            image_embeds = self.vision_model(
                pixel_values, return_dict=True
            ).last_hidden_state

            # step 2: forward the query tokens through the QFormer,
            # using the image embeddings for cross-attention
            # (num_videos, time * vision_seq_len, vision_hidden_size)
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
            # (num_videos, num_query_tokens, qformer_hidden_size)
            query_output = query_outputs.last_hidden_state

            # step 3: project the qformer tokens to the language model space
            num_videos = pixel_values.size(0)
            # (num_videos * num_query_tokens, hidden)
            prompt_video_features = self.language_projection(
                query_output.view(num_videos * self.config.num_query_tokens, -1)
            )
        # (batch, prompt_seq_len, hidden)
        prompt_input_embeds = self.language_model.get_input_embeddings()(
            prompt_input_ids.to(self.language_model.device)
        )
        if prompt_video_features is not None:
            prompt_input_embeds[prompt_video_input_mask] = prompt_video_features
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)
        prompt_outputs = self.language_model(
            inputs_embeds=prompt_input_embeds,
            attention_mask=prompt_attention_mask,
            return_dict=True,
            use_cache=True,
        )

        # step 4: calculate the mean log likelihood using the cached results from step 3
        num_classes = class_input_ids.size(0)
        if class_batch_size is None:
            class_batch_size = num_classes
        mean_class_log_likelihoods: list[torch.Tensor] = []
        batch = prompt_input_ids.size(0)
        for i in range(0, num_classes, class_batch_size):
            # (batch, class_batch_size)
            mean_log_likelihood = self._calc_class_log_likelihood(
                batch,
                class_input_ids[i : i + class_batch_size],
                prompt_outputs.logits,
                prompt_outputs.past_key_values,
                prompt_attention_mask,
                class_attention_mask=None
                if class_attention_mask is None
                else class_attention_mask[i : i + class_batch_size],
            )
            mean_class_log_likelihoods.append(mean_log_likelihood)
        return torch.cat(mean_class_log_likelihoods, dim=1)

    def _calc_class_log_likelihood(
        self,
        batch: int,
        class_input_ids: torch.Tensor,
        prompt_logits: torch.Tensor,
        prompt_past_key_values: tuple[tuple[torch.Tensor]],
        prompt_attention_mask: torch.Tensor,
        class_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_classes, _ = class_input_ids.size()
        # (batch * num_classes, class_seq_len)
        batch_class_input_ids = (
            class_input_ids.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * num_classes, -1)
        )
        if class_attention_mask is None:
            class_attention_mask = torch.ones_like(class_input_ids)
        # (batch * num_classes, prompt_seq_len + class_seq_len)
        batch_class_attention_mask = torch.cat(
            [
                # (batch * num_classes, prompt_seq_len)
                prompt_attention_mask.repeat_interleave(num_classes, dim=0),
                # (batch * num_classes, class_seq_len)
                class_attention_mask.unsqueeze(0)
                .expand(batch, -1, -1)
                .reshape(batch * num_classes, -1),
            ],
            dim=1,
        )
        # (batch * num_classes, num_heads, prompt_seq_len, hidden_per_head)
        batch_past_key_values = tuple(
            tuple(kv.repeat_interleave(num_classes, dim=0) for kv in layer_kv)
            for layer_kv in prompt_past_key_values
        )
        outputs = self.language_model(
            input_ids=batch_class_input_ids,
            attention_mask=batch_class_attention_mask,
            past_key_values=batch_past_key_values,
            return_dict=True,
        )

        # Use the last logits from the prompt to shift class logits
        # (batch * num_classes, class_seq_len, hidden)
        shift_logits = torch.cat(
            [
                prompt_logits[:, -1:].repeat_interleave(num_classes, dim=0),
                outputs.logits[:, :-1],
            ],
            dim=1,
        )
        # (num_classes, class_seq_len)
        labels = class_input_ids.clone()
        labels[class_attention_mask == 0] = -100
        # (batch * num_classes, class_seq_len)
        labels = (
            labels.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * num_classes, -1)
            .to(shift_logits.device)
        )

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # (batch, num_classes, num_tokens - 1)
        loss = loss_fct(
            shift_logits.view(-1, self.config.text_config.vocab_size),
            labels.reshape(-1),
        ).view(batch, num_classes, -1)

        # return the mean log likelihood
        # (batch, num_classes)
        sum_log_likelihood = -loss.sum(dim=-1)
        # (batch, num_classes)
        class_lengths = class_attention_mask.sum(dim=-1).unsqueeze(0).expand(batch, -1)
        return sum_log_likelihood / class_lengths
