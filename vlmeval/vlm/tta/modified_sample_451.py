from transformers.generation.utils import *
import torch
import torch.nn as nn
import os
from typing import Optional, Union


# transformers==4.51.3
# Only the token selection part of the _sample method was modified
def _modified_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    # DEBUG: Check if this modified _sample function is being called
    print("******* DEBUG: _modified_sample function called! *******")
    print(f"DEBUG: do_sample = {generation_config.do_sample}")
    print(
        f"DEBUG: token_selection_aggregation_method = {getattr(self, 'token_selection_aggregation_method', 'NOT_SET')}"
    )
    print("********************************************************")
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = (
            model_kwargs["past_key_values"].is_compileable
            and self._supports_static_cache
        )
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda"
            or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # Token Selection (ours)
        if self.token_selection_aggregation_method in [
            "original",
            "clip_synthesizer",
            "mllm_synthesizer",
            "mllm_synthesizer_with_input",
            "orm_best_of_n",
            "prm_best_of_n",
            "prm_best_of_n_weighting",
            "average_features_early_layer",
        ]:
            # Token Selection (original - no aggregation)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
        elif self.token_selection_aggregation_method == "average":
            # Token Selection (ours - simple averaging)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            probs = probs.mean(dim=0, keepdim=True)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            next_tokens = next_tokens.repeat(batch_size)
        elif self.token_selection_aggregation_method == "average_before_softmax":
            # Token Selection (ours - simple averaging)
            averaged_scores = next_token_scores.mean(dim=0, keepdim=True)
            probs = nn.functional.softmax(averaged_scores, dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            next_tokens = next_tokens.repeat(batch_size)
        elif self.token_selection_aggregation_method == "majority":
            # Token Selection (ours - majority voting)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            next_tokens = torch.mode(next_tokens, dim=0).values
            next_tokens = next_tokens.repeat(batch_size)
        elif self.token_selection_aggregation_method == "ewm":
            # Token Selection (ours - entropy-based weighted sampler using reciprocal softmax normalization)
            probs = nn.functional.softmax(
                next_token_scores, dim=-1
            )  # Shape: (num_augmentations, vocab_size)

            # Calculate entropy for each augmentation
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-12), dim=-1
            )  # Shape: (num_augmentations,)

            # Calculate weights based on reciprocal softmax of entropy
            exp_entropy = torch.exp(entropy)
            softmax_entropy = exp_entropy / exp_entropy.sum(dim=0)
            weights = 1 / softmax_entropy
            weights = weights / weights.sum(dim=0)  # Normalize weights to sum to 1

            # Aggregate probabilities using the weights
            weighted_probs = torch.sum(
                probs * weights.unsqueeze(-1), dim=0, keepdim=True
            )  # Shape: (1, vocab_size)

            # Select the next token based on aggregated probabilities
            if do_sample:
                next_tokens = torch.multinomial(weighted_probs, num_samples=1).squeeze(
                    1
                )
            else:
                next_tokens = torch.argmax(weighted_probs, dim=-1)

            next_tokens = next_tokens.repeat(batch_size)
        elif self.token_selection_aggregation_method == "mostconfident":
            # Token Selection (ours - most confident token across different augmentations)
            probs = nn.functional.softmax(
                next_token_scores, dim=-1
            )  # Shape: (num_augmentations, vocab_size)

            # Find the maximum probability across augmentations for each token in the vocabulary
            max_probs, _ = torch.max(
                probs, dim=0, keepdim=True
            )  # Shape: (1, vocab_size)

            # Select the next token based on the maximum probabilities
            if do_sample:
                next_tokens = torch.multinomial(max_probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(max_probs, dim=-1)

            # Repeat tokens for the batch size
            next_tokens = next_tokens.repeat(batch_size)

        elif (
            self.token_selection_aggregation_method
            == "prm_best_of_n_weighting_generate_with_weighting"
        ):
            # Token Selection (ours - weighted averaging)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            weights = torch.nn.functional.softmax(
                torch.tensor(self.weighting_vector).to(probs.device), dim=0
            )  # Shape: (num_augmentations,)
            probs = probs * weights.unsqueeze(
                -1
            )  # Shape: (num_augmentations, vocab_size)
            probs = probs.mean(dim=0, keepdim=True)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            next_tokens = next_tokens.repeat(batch_size)

        elif self.token_selection_aggregation_method == "learned":
            with torch.enable_grad():
                device = next_token_scores.device  # Ensure all tensors use this device

                # Create or ensure the aggregation_weights parameter is registered and requires grad.
                if not hasattr(self, "aggregation_weights"):
                    num_augmentations = next_token_scores.shape[0]
                    self.aggregation_weights = torch.nn.Parameter(
                        torch.ones(num_augmentations, device=device) / num_augmentations
                    )
                    # Use adaptive learning rate based on sequence position
                    initial_lr = getattr(self, "tta_learning_rate", 1e-2)
                    self._tta_optimizer = torch.optim.AdamW(
                        [self.aggregation_weights], lr=initial_lr, weight_decay=1e-4
                    )
                    # Track optimization steps for this token
                    self._token_step_count = 0
                else:
                    self.aggregation_weights = self.aggregation_weights.to(device)
                    if not self.aggregation_weights.requires_grad:
                        self.aggregation_weights.requires_grad_()

                # Compute probabilities from the detached scores.
                probs = torch.nn.functional.softmax(
                    next_token_scores.detach(), dim=-1
                ).to(device)

                # Multi-step optimization for better convergence
                num_tta_steps = getattr(self, "num_tta_steps", 20)

                for step in range(num_tta_steps):
                    # Compute learned weights (this will be differentiable).
                    learned_weights = torch.nn.functional.softmax(
                        self.aggregation_weights, dim=0
                    )
                    learned_weights = learned_weights.to(device)
                    print(
                        f"TTA Step {step + 1}/{num_tta_steps}, learned_weights: {learned_weights}"
                    )

                    # Compute weighted probabilities: only learned_weights should be differentiable.
                    weighted_probs = torch.sum(
                        probs * learned_weights.unsqueeze(-1), dim=0, keepdim=True
                    )

                    # Compute marginal entropy with additional regularization
                    marginal_entropy = -torch.sum(
                        weighted_probs * torch.log(weighted_probs + 1e-12)
                    )

                    # Add weight regularization to prevent overfitting to single augmentation
                    # weight_regularization = getattr(self, "weight_reg_strength", 0.01)
                    # reg_loss = weight_regularization * torch.sum(learned_weights**2)
                    total_loss = marginal_entropy  # + reg_loss

                    self._tta_optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self.aggregation_weights, max_norm=1.0
                    )

                    self._tta_optimizer.step()

                # Use final optimized weights for token selection
                final_learned_weights = torch.nn.functional.softmax(
                    self.aggregation_weights, dim=0
                )
                final_weighted_probs = torch.sum(
                    probs * final_learned_weights.unsqueeze(-1), dim=0, keepdim=True
                )

                # Select next tokens using aggregated probabilities.
                if do_sample:
                    next_tokens = torch.multinomial(
                        final_weighted_probs, num_samples=1
                    ).squeeze(1)
                else:
                    next_tokens = torch.argmax(final_weighted_probs, dim=-1)

                next_tokens = next_tokens.to(device).repeat(batch_size)
        elif self.token_selection_aggregation_method == "learned_model":
            raise NotImplementedError(
                "YOU SHOULD NOT SEE THIS! The 'learned_model' token selection aggregation method is not implemented here. Outside of the scope of this code: learned_model_generate()"
            )
        elif self.token_selection_aggregation_method == "learned_embedding":
            raise NotImplementedError(
                "YOU SHOULD NOT SEE THIS!  The 'learned_embedding' token selection aggregation method is not implemented HERE."
            )
        else:
            raise ValueError(
                f"Invalid token selection aggregation method: {self.token_selection_aggregation_method}"
            )
        ###########

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
