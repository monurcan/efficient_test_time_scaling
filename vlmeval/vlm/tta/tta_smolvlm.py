try:
    import unsloth
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except (ImportError, AttributeError) as e:
    print(
        "************************************************"
        "Warning: unsloth/trl libraries not available. 'learned_model' aggregation method will not work.",
        e,
        "************************************************",
    )


import contextlib
import copy
import os  # Add import for os
import time
import types
from functools import partial, partialmethod

import outlines
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)
import torchvision.utils as vutils
from torchvision import transforms

from ...dataset import DATASET_MODALITY, DATASET_TYPE
from ...smp import *
from ..smolvlm import SmolVLM2
from .image_augment import ImageAugment
from .modified_sample_451 import _modified_sample
from .text_augment import TextAugment


class TTAugAdapter_SmolVLM2(SmolVLM2):
    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        print("================================")
        print(f"Initializing TTAugAdapter_SmolVLM2")
        print(f"Adapter Args: {adapter_args}")
        print("================================")

        for key, value in adapter_args.items():
            setattr(self, key, value)

        # Load model using unsloth if using learned_model method
        if self.token_selection_aggregation_method == "learned_model":
            self.load_model_with_unsloth()
        else:
            super().__init__(**model_args)

        self.text_augment = TextAugment(
            n_augmentations=self.number_of_versions,
            **text_aug_args,
        )

        self.image_augment = ImageAugment(
            n_augmentations=self.number_of_versions, **image_aug_args
        )

        if self.token_selection_aggregation_method == "learned_model":
            print("! Using learned_model method with Unsloth")
            self.model_unsloth.token_selection_aggregation_method = (
                self.token_selection_aggregation_method
            )

            self.model_unsloth._sample = types.MethodType(
                _modified_sample, self.model_unsloth
            )
        elif self.token_selection_aggregation_method in [
            "answer_level_temperature_mllm_selector",
            "answer_level_temperature_majority_vote",
            "answer_level_temperature_confidence_scores",
            "answer_level_temperature_mllm_synthesizer",
            "answer_level_greedy_mllm_selector",
            "answer_level_greedy_majority_vote",
            "answer_level_greedy_confidence_scores",
            "answer_level_greedy_mllm_synthesizer",
        ]:
            print("! Using an answer_level method")
        else:
            self.model.token_selection_aggregation_method = (
                self.token_selection_aggregation_method
            )

            self.model._sample = types.MethodType(_modified_sample, self.model)

    def create_message_versions(self, message):
        versions = [copy.deepcopy(message) for _ in range(self.number_of_versions)]

        def split_options(text):
            text = text.strip()
            if not text or len(text.split()) <= 2:
                return None, None
            if "Options:" in text:
                main, opts = text.split("Options:", 1)
                return main.strip(), "Options:" + opts
            return text, ""

        for msg_idx, msg in enumerate(message):
            # Handle top-level text
            if msg.get("type") == "text":
                base, opts = split_options(msg["value"])
                if base is None:
                    continue
                for ver_idx, paraphrase in enumerate(self.text_augment(base)):
                    versions[ver_idx][msg_idx]["value"] = paraphrase + opts

            # Handle nested content blocks
            content = msg.get("content", [])
            for item_idx, item in enumerate(content):
                if item.get("type") != "text":
                    continue
                base, opts = split_options(item["value"])
                if base is None:
                    continue
                for ver_idx, paraphrase in enumerate(self.text_augment(base)):
                    versions[ver_idx][msg_idx]["content"][item_idx]["value"] = (
                        paraphrase + opts
                    )

        return versions

    def build_prompt_cases(self, message, dataset=None):
        # augments the text but not images!
        message_versions = self.create_message_versions(message)

        formatted_messages = []
        for message_version in message_versions:
            formatted_message, formatted_images = super().build_prompt_cases(
                message_version, dataset
            )
            formatted_messages.append(formatted_message)

        return formatted_messages, formatted_images

    def load_model_with_unsloth(self):
        """Load model using unsloth for efficient training."""
        print("Loading model with Unsloth...")

        model_path = self.model_args.get(
            "model_path", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )

        # Load model and processor using unsloth with new options
        self.model_unsloth, self.processor = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=True,
            # use_gradient_checkpointing="unsloth",
            use_gradient_checkpointing=False,
        )

        print(f"Unsloth model loaded successfully: {type(self.model_unsloth)}")

        # Store initial model state for resetting weights
        self._store_initial_model_state()

        # Setup unsloth adapter immediately
        print("Setting up Unsloth adapter for SmolVLM")
        # Get the patchable model with LoRA
        # self.model_unsloth.enable_input_require_grads()
        # self.model_unsloth = FastVisionModel.get_peft_model(
        #     self.model_unsloth,
        #     finetune_vision_layers=True,
        #     finetune_language_layers=True,
        #     finetune_attention_modules=True,
        #     finetune_mlp_modules=True,
        #     r=getattr(self, "lora_rank", 16),
        #     lora_alpha=getattr(self, "lora_alpha", 16),
        #     lora_dropout=getattr(self, "lora_dropout", 0),
        #     bias="none",
        #     random_state=3407,
        #     use_rslora=False,
        #     loftq_config=None,
        #     # target_modules="all-linear",
        #     # modules_to_save=[
        #     #     "lm_head",
        #     #     "embed_tokens",
        #     # ],
        # )

    def _store_initial_model_state(self):
        """Store the initial model state for resetting weights."""
        print("Storing initial model state for weight reset...")
        self._initial_model_state = {}

        # Store the state dict of all trainable parameters
        for name, param in self.model_unsloth.named_parameters():
            if param.requires_grad:
                self._initial_model_state[name] = param.data.clone().detach()

        print(f"Stored {len(self._initial_model_state)} trainable parameter states")

    def verify_model_reset(self):
        """Verify that model weights have been properly reset (for debugging)."""
        if not hasattr(self, "_initial_model_state"):
            print("No initial model state stored - cannot verify reset")
            return False

        differences = 0
        total_params = 0

        for name, param in self.model_unsloth.named_parameters():
            if name in self._initial_model_state and param.requires_grad:
                total_params += 1
                if not torch.equal(param.data, self._initial_model_state[name]):
                    differences += 1

        print(
            f"Model reset verification: {differences}/{total_params} parameters differ from initial state"
        )
        return differences == 0

    def train_with_pseudolabels(self, pseudolabel_data):
        """Train model using pseudolabel data with original aggregation."""
        print("Training model with pseudolabels...")
        with torch.enable_grad():
            # Save original aggregation method
            original_method = self.model_unsloth.token_selection_aggregation_method

            # Set to "original" for training (no augmentation aggregation)
            self.model_unsloth.token_selection_aggregation_method = "original"

            # Ensure the model is properly set up for training with unsloth
            FastVisionModel.for_training(self.model_unsloth)

            # Ensure model is in training mode after weight reset
            self.model_unsloth.train()

            # Setup trainer for vision model training - following unsloth documentation
            trainer = SFTTrainer(
                model=self.model_unsloth,
                train_dataset=pseudolabel_data,
                processing_class=self.processor.tokenizer,
                data_collator=UnslothVisionDataCollator(
                    self.model_unsloth, self.processor
                ),
                args=SFTConfig(
                    per_device_train_batch_size=getattr(
                        self, "pseudolabel_batch_size", 64
                    ),
                    gradient_accumulation_steps=getattr(
                        self, "gradient_accum_steps", 2
                    ),
                    # gradient_checkpointing=True,
                    # gradient_checkpointing_kwargs={"use_reentrant": False},
                    # max_grad_norm=0.3,
                    # warmup_ratio=0.03,
                    warmup_steps=5,
                    max_steps=getattr(self, "pseudolabel_training_steps", 6),
                    learning_rate=getattr(self, "pseudolabel_learning_rate", 2e-6),
                    logging_steps=1,
                    save_strategy="no",  # Don't save checkpoints
                    optim="adamw_torch_fused",
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    seed=3407,
                    output_dir="/tmp/pseudolabel_training",
                    report_to="none",
                    # Required for vision finetuning - following unsloth docs
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=2048,
                ),
            )

            # Train
            trainer_stats = trainer.train()
            print(
                f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds"
            )

            # Clear memory
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Restore original aggregation method
            self.model_unsloth.token_selection_aggregation_method = original_method

    def create_pseudolabel_dataset_from_tta(
        self, formatted_messages, images_augmented, return_final_response, **inputs
    ):
        """Create pseudolabel dataset from TTA inputs using average aggregation."""
        print(f"Creating pseudolabels from {len(formatted_messages)} TTA samples...")

        # Save original aggregation method
        original_method = self.model_unsloth.token_selection_aggregation_method

        # Set to "average" for TTA inference to generate pseudolabels
        self.model_unsloth.token_selection_aggregation_method = "average"

        # Generate response using TTA (average aggregation)
        with torch.no_grad():
            # Try with the full batch first
            generated_ids = self.model_unsloth.generate(**inputs)
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].size(1) :],
                skip_special_tokens=True,
            )

        self.model_unsloth.token_selection_aggregation_method = original_method

        final_response = generated_text[0].strip()

        if return_final_response:
            return final_response

        print(f"Generated pseudolabel response: {final_response[:100]}...")

        # DON'T set inference mode here - we need to keep the model trainable
        # We'll just use torch.no_grad() for this inference step
        pseudolabel_data = []

        # Create training samples for each augmented input with the same target response
        for formatted_msg, img in zip(formatted_messages, images_augmented):
            # Extract the question part (before "Assistant:")
            question_part = formatted_msg.split("Assistant:")[0].strip()
            if question_part.endswith("<end_of_utterance>"):
                question_part = question_part[: -len("<end_of_utterance>")].strip()

            # Remove the "User:" prefix if present for clean text
            if question_part.startswith("<|im_start|>User:"):
                question_text = question_part[len("<|im_start|>User:") :].strip()
            else:
                question_text = question_part

            # Remove any image placeholders
            question_text = question_text.replace("<image>", "").strip()

            # Create conversation in unsloth format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_text},
                        {"type": "image", "image": img[0]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_response}],
                },
            ]

            pseudolabel_data.append({"messages": conversation})

        print(f"Created {len(pseudolabel_data)} pseudolabel samples")
        print("\n\n\n\n*************", pseudolabel_data, "\n\n\n\n*************")

        return pseudolabel_data

    def learned_model_generate(self, formatted_messages, images_augmented):
        """Generate using iterative pseudolabel training with unsloth."""
        print("Starting pseudolabel-based generation...")

        # Reset learned weights/adapter weights for each new question
        # Reset model weights to initial state if using learned_model method
        if hasattr(self, "_initial_model_state") and getattr(
            self, "reset_model_weights", True
        ):
            print("Resetting model weights to initial state...")

            with torch.no_grad():
                for name, param in self.model_unsloth.named_parameters():
                    if name in self._initial_model_state and param.requires_grad:
                        param.data.copy_(self._initial_model_state[name])

            print("Model weights reset complete")

            # Clear any cached optimizer states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Optional: verify reset worked (for debugging)
        if getattr(self, "verify_reset", False):
            self.verify_model_reset()

        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=formatted_messages,
            images=images_augmented,
            return_tensors="pt",
            padding=True,
        ).to(self.model_unsloth.device, dtype=self.model_unsloth.dtype)

        max_iterations = getattr(self, "pseudolabel_iterations", 3)

        for iteration in range(max_iterations):
            print(f"Pseudolabel iteration {iteration + 1}/{max_iterations}")

            # Create pseudolabel dataset using current model state
            pseudolabel_data = self.create_pseudolabel_dataset_from_tta(
                formatted_messages, images_augmented, False, **inputs
            )

            # Train model with pseudolabels (except on last iteration)
            if iteration < max_iterations - 1:
                self.train_with_pseudolabels(pseudolabel_data)

            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate final response using the trained model
        # Only set inference mode for the final generation
        FastVisionModel.for_inference(self.model_unsloth)

        generated_text = self.create_pseudolabel_dataset_from_tta(
            formatted_messages, images_augmented, True, **inputs
        )

        print(
            "***********************************************FINAL ANSWER***********************************************",
            generated_text,
            "*********************<FINAL ANSWER>**************************",
        )

        # Return aggregated response (take first if multiple)
        return generated_text if generated_text else ""

    def register_feature_averaging_hook(self):
        # Get the layer index from adapter_args
        k = getattr(self, "average_features_early_layer", None)
        if k is None:
            raise ValueError("average_features_early_layer must be specified!")

        # Define the collapse hook
        def collapse_hook(module, inputs, output):
            print(f"Hook called on layer {k}")
            print(f"Output type: {type(output)}")

            if isinstance(output, tuple):
                print(f"Tuple length: {len(output)}")
                hidden_states = output[0]
                print(f"Hidden states shape before: {hidden_states.shape}")

                # Only average if we have more than 1 sample in batch
                if hidden_states.shape[0] > 1:
                    # Average across batch dimension and expand back
                    averaged = hidden_states.mean(dim=0, keepdim=True)
                    collapsed_hidden = averaged.expand_as(hidden_states)
                    print(f"Hidden states shape after: {collapsed_hidden.shape}")

                    # Return the modified tuple
                    return (collapsed_hidden,) + output[1:]
                else:
                    print("Single batch item, no averaging needed")
                    return output
            else:
                # Single tensor output
                print(f"Hidden states shape before: {output.shape}")
                if output.shape[0] > 1:
                    averaged = output.mean(dim=0, keepdim=True)
                    collapsed = averaged.expand_as(output)
                    print(f"Hidden states shape after: {collapsed.shape}")
                    return collapsed
                else:
                    print("Single batch item, no averaging needed")
                    return output

        # Register hook on the k-th text model layer
        hook_handle = self.model.model.text_model.layers[k].register_forward_hook(
            collapse_hook
        )

        return hook_handle

    def answer_level_temperature_mllm_selector_generate(
        self, formatted_message, image, **inputs
    ):
        """
        Answer-level aggregation with MLLM selector via multinomial + temperature sampling.
        First generate diverse responses with the model.
        Then select one of them using the model as selector (via structured generation).
        """

        # First repeat the original input to create batch
        # Take the last item from each tensor and repeat it to match the batch size
        repeated_inputs = {}

        for k, v in inputs.items():
            # Repeat to create batch of same size
            repeated_inputs[k] = (
                v[-1].unsqueeze(0).repeat(v.size(0), *([1] * (len(v.shape) - 1)))
            )

        kwargs_multinomial_temp = {
            **self.kwargs,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        generated_ids = self.model.generate(
            **repeated_inputs, **kwargs_multinomial_temp
        )
        generated_text = self.processor.batch_decode(
            generated_ids[:, repeated_inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(kwargs_multinomial_temp)
        print(generated_text, "\n************************")

        # Selector
        if not hasattr(self, "structured_selector"):
            self.structured_selector = outlines.generate.choice(
                outlines.models.transformers_vision(
                    self.model_args["model_path"],
                    model_class=SmolVLMForConditionalGeneration,
                    device="cuda",
                ),
                [str(i) for i in range(self.number_of_versions)],
                sampler=outlines.samplers.greedy(),
            )

        generated_responses_formatted = "\n".join(
            f"Answer {i}: {text.strip()}" for i, text in enumerate(generated_text)
        )
        selector_prompt = (
            f"{formatted_message.replace('User:', 'User: Question: ').replace('Assistant:', '').replace('<end_of_utterance>', '')}\n"
            f"Different people answered this question in different ways. Select the best response from these candidate answers:\n"
            f"{generated_responses_formatted}\n"
            f"Just return the index of the best response. Return an integer between 0 and {self.number_of_versions - 1}.<end_of_utterance>\nAssistant:"
        )

        print(selector_prompt)

        selected_index = int(self.structured_selector(selector_prompt, image))

        print(f"Selected index: {selected_index}")

        # Return the selected response
        generated_text = generated_text[selected_index]

        return generated_text.strip()

    def answer_level_temperature_majority_vote_generate(
        self, formatted_message, image, **inputs
    ):
        """
        Answer-level aggregation with majority voting via multinomial + temperature sampling.
        First generate diverse responses with the model.
        Then select one of them via majority vote.
        """

        # First repeat the original input to create batch
        # Take the last item from each tensor and repeat it to match the batch size
        repeated_inputs = {}

        for k, v in inputs.items():
            # Repeat to create batch of same size
            repeated_inputs[k] = (
                v[-1].unsqueeze(0).repeat(v.size(0), *([1] * (len(v.shape) - 1)))
            )

        kwargs_multinomial_temp = {
            **self.kwargs,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        generated_ids = self.model.generate(
            **repeated_inputs, **kwargs_multinomial_temp
        )
        generated_text = self.processor.batch_decode(
            generated_ids[:, repeated_inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(kwargs_multinomial_temp)
        print(generated_text, "\n************************")

        # First preprocess answers to normalize them
        processed_answers = [
            text.strip()
            .lower()
            .replace("answer: ", "")
            .replace("the answer is ", "")
            .strip()
            for text in generated_text
        ]
        print("Processed answers for majority vote:", processed_answers)
        # Majority vote
        from collections import Counter

        answer_counts = Counter(processed_answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        print(f"Most common answer: '{most_common_answer}' with count {count}")
        generated_text = most_common_answer

        return generated_text.strip()

    def answer_level_temperature_confidence_scores_generate(
        self, formatted_message, image, **inputs
    ):
        # First repeat the original input to create batch
        # Take the last item from each tensor and repeat it to match the batch size
        repeated_inputs = {}

        for k, v in inputs.items():
            # Repeat to create batch of same size
            repeated_inputs[k] = (
                v[-1].unsqueeze(0).repeat(v.size(0), *([1] * (len(v.shape) - 1)))
            )

        kwargs_multinomial_temp = {
            **self.kwargs,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        generated_dict = self.model.generate(
            **repeated_inputs,
            **kwargs_multinomial_temp,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = generated_dict.sequences

        generated_text = self.processor.batch_decode(
            generated_ids[:, repeated_inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(kwargs_multinomial_temp)
        print(generated_text, "\n************************")

        # Selector based on sum of confidence scores
        # scores = generated_dict.scores  # tuple(len = seq_len), each (batch, vocab_size)
        # batch_size = generated_ids.size(0)
        # input_len = inputs["input_ids"].size(1)
        # gen_len = len(scores)

        # # Stack scores into tensor: (gen_len, batch, vocab_size)
        # score_tensor = torch.stack(scores, dim=0)

        # # Log-softmax across vocab
        # log_probs = torch.nn.functional.log_softmax(
        #     score_tensor, dim=-1
        # )  # (gen_len, batch, vocab)

        # # Gather the log-probs of the chosen tokens
        # chosen_ids = generated_ids[:, input_len:]  # (batch, gen_len)
        # chosen_ids = chosen_ids.transpose(0, 1).unsqueeze(-1)  # (gen_len, batch, 1)
        # token_logprobs = torch.gather(log_probs, 2, chosen_ids).squeeze(
        #     -1
        # )  # (gen_len, batch)

        # # Sum over sequence → sequence log-likelihood
        # sequence_scores = token_logprobs.sum(dim=0)  # (batch,)

        # Compute per-token transition scores (log-probabilities)
        transition_scores = self.model.compute_transition_scores(
            sequences=generated_dict.sequences,
            scores=generated_dict.scores,
            normalize_logits=True,
        )  # (batch, generated_sequence_length)

        # Sum log-probs for sequence-level score
        sequence_scores = transition_scores.sum(dim=1)  # (batch,)

        # Pick best
        best_idx = sequence_scores.argmax().item()
        best_answer = generated_text[best_idx]

        # print("confidence scores:")
        # print(transition_scores)
        # print("sequence scores:")
        # print(sequence_scores)
        # print("shapes:")
        # print(
        #     transition_scores.shape,
        #     sequence_scores.shape,
        #     generated_ids[:, inputs["input_ids"].size(1) :].shape,
        # )
        # print(f"Best index: {best_idx}")

        return best_answer.strip()

    def answer_level_temperature_mllm_synthesizer_generate(
        self, formatted_message, image, **inputs
    ):
        # First repeat the original input to create batch
        # Take the last item from each tensor and repeat it to match the batch size
        repeated_inputs = {}

        for k, v in inputs.items():
            # Repeat to create batch of same size
            repeated_inputs[k] = (
                v[-1].unsqueeze(0).repeat(v.size(0), *([1] * (len(v.shape) - 1)))
            )

        kwargs_multinomial_temp = {
            **self.kwargs,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        generated_ids = self.model.generate(
            **repeated_inputs, **kwargs_multinomial_temp
        )
        generated_text = self.processor.batch_decode(
            generated_ids[:, repeated_inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(kwargs_multinomial_temp)
        print(generated_text, "\n************************")

        generated_responses_formatted = "\n".join(
            f"Answer {i}: {text.strip()}" for i, text in enumerate(generated_text)
        )
        selector_prompt = (
            f"{formatted_message.replace('User:', 'User: Question: ').replace('Assistant:', '').replace('<end_of_utterance>', '')}\n"
            f"Different people answered this question in different ways. Combine these responses into a single, coherent and accurate answer:\n"
            f"{generated_responses_formatted}\n"
            f"Just return the final answer.<end_of_utterance>\nAssistant:"
        )

        print(selector_prompt)

        inputs = self.processor(
            text=selector_prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )
        print(generated_text, "\n************************")

        return generated_text[-1].strip()

    def answer_level_greedy_mllm_selector_generate(
        self, formatted_message, image, **inputs
    ):
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(generated_text, "\n************************")

        # Selector
        if not hasattr(self, "structured_selector"):
            self.structured_selector = outlines.generate.choice(
                outlines.models.transformers_vision(
                    self.model_args["model_path"],
                    model_class=SmolVLMForConditionalGeneration,
                    device="cuda",
                ),
                [str(i) for i in range(self.number_of_versions)],
                sampler=outlines.samplers.greedy(),
            )

        generated_responses_formatted = "\n".join(
            f"Answer {i}: {text.strip()}" for i, text in enumerate(generated_text)
        )
        selector_prompt = (
            f"{formatted_message.replace('User:', 'User: Question: ').replace('Assistant:', '').replace('<end_of_utterance>', '')}\n"
            f"Different people answered this question in different ways. Select the best response from these candidate answers:\n"
            f"{generated_responses_formatted}\n"
            f"Just return the index of the best response. Return an integer between 0 and {self.number_of_versions - 1}.<end_of_utterance>\nAssistant:"
        )

        print(selector_prompt)

        selected_index = int(self.structured_selector(selector_prompt, image))

        print(f"Selected index: {selected_index}")

        # Return the selected response
        generated_text = generated_text[selected_index]

        return generated_text.strip()

    def answer_level_greedy_majority_vote_generate(
        self, formatted_message, image, **inputs
    ):
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(generated_text, "\n************************")

        # First preprocess answers to normalize them
        processed_answers = [
            text.strip()
            .lower()
            .replace("answer: ", "")
            .replace("the answer is ", "")
            .strip()
            for text in generated_text
        ]
        print("Processed answers for majority vote:", processed_answers)
        # Majority vote
        from collections import Counter

        answer_counts = Counter(processed_answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        print(f"Most common answer: '{most_common_answer}' with count {count}")
        generated_text = most_common_answer

        return generated_text.strip()

    def answer_level_greedy_mllm_synthesizer_generate(
        self, formatted_message, image, **inputs
    ):
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(generated_text, "\n************************")

        generated_responses_formatted = "\n".join(
            f"Answer {i}: {text.strip()}" for i, text in enumerate(generated_text)
        )
        selector_prompt = (
            f"{formatted_message.replace('User:', 'User: Question: ').replace('Assistant:', '').replace('<end_of_utterance>', '')}\n"
            f"Different people answered this question in different ways. Combine these responses into a single, coherent and accurate answer:\n"
            f"{generated_responses_formatted}\n"
            f"Just return the final answer.<end_of_utterance>\nAssistant:"
        )

        print(selector_prompt)

        inputs = self.processor(
            text=selector_prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )
        print(generated_text, "\n************************")

        return generated_text[-1].strip()

    def answer_level_greedy_confidence_scores_generate(
        self, formatted_message, image, **inputs
    ):
        """
        Answer-level aggregation with confidence scores. Select the answer with the highest log prob, arg max log p (x)
        """
        generated_dict = self.model.generate(
            **inputs,
            **self.kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = generated_dict.sequences
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

        print(generated_text, "\n************************")

        # Selector based on sum of confidence scores
        # scores = generated_dict.scores  # tuple(len = seq_len), each (batch, vocab_size)
        # batch_size = generated_ids.size(0)
        # input_len = inputs["input_ids"].size(1)
        # gen_len = len(scores)

        # # Stack scores into tensor: (gen_len, batch, vocab_size)
        # score_tensor = torch.stack(scores, dim=0)

        # # Log-softmax across vocab
        # log_probs = torch.nn.functional.log_softmax(
        #     score_tensor, dim=-1
        # )  # (gen_len, batch, vocab)

        # # Gather the log-probs of the chosen tokens
        # chosen_ids = generated_ids[:, input_len:]  # (batch, gen_len)
        # chosen_ids = chosen_ids.transpose(0, 1).unsqueeze(-1)  # (gen_len, batch, 1)
        # token_logprobs = torch.gather(log_probs, 2, chosen_ids).squeeze(
        #     -1
        # )  # (gen_len, batch)

        # # Sum over sequence → sequence log-likelihood
        # sequence_scores = token_logprobs.sum(dim=0)  # (batch,)

        # Compute per-token transition scores (log-probabilities)
        transition_scores = self.model.compute_transition_scores(
            sequences=generated_dict.sequences,
            scores=generated_dict.scores,
            normalize_logits=True,
        )  # (batch, generated_sequence_length)

        # Sum log-probs for sequence-level score
        sequence_scores = transition_scores.sum(dim=1)  # (batch,)

        # Pick best
        best_idx = sequence_scores.argmax().item()
        best_answer = generated_text[best_idx]

        # print("confidence scores:")
        # print(transition_scores)
        # print("sequence scores:")
        # print(sequence_scores)
        # print("shapes:")
        # print(
        #     transition_scores.shape,
        #     sequence_scores.shape,
        #     generated_ids[:, inputs["input_ids"].size(1) :].shape,
        # )
        # print(f"Best index: {best_idx}")

        return best_answer.strip()

    def reset_learned_aggregation_weights(self):
        """Reset learned aggregation weights and model weights for each new question."""
        # Reset aggregation-related attributes
        if hasattr(self.model, "aggregation_weights"):
            delattr(self.model, "aggregation_weights")
        if hasattr(self.model, "_tta_optimizer"):
            delattr(self.model, "_tta_optimizer")
        if hasattr(self.model, "_token_step_count"):
            delattr(self.model, "_token_step_count")

    def generate_inner_helper(
        self, message, formatted_messages, images_augmented, dataset=None
    ):
        if self.token_selection_aggregation_method == "learned_model":
            return self.learned_model_generate(formatted_messages, images_augmented)

        # Process text and images directly
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=formatted_messages,
            images=images_augmented,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        if (
            self.token_selection_aggregation_method
            == "answer_level_temperature_mllm_selector"
        ):
            return self.answer_level_temperature_mllm_selector_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_temperature_majority_vote"
        ):
            return self.answer_level_temperature_majority_vote_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_temperature_confidence_scores"
        ):
            return self.answer_level_temperature_confidence_scores_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_temperature_mllm_synthesizer"
        ):
            return self.answer_level_temperature_mllm_synthesizer_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_greedy_mllm_selector"
        ):
            return self.answer_level_greedy_mllm_selector_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_greedy_majority_vote"
        ):
            return self.answer_level_greedy_majority_vote_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_greedy_mllm_synthesizer"
        ):
            return self.answer_level_greedy_mllm_synthesizer_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        if (
            self.token_selection_aggregation_method
            == "answer_level_greedy_confidence_scores"
        ):
            return self.answer_level_greedy_confidence_scores_generate(
                formatted_messages[-1], images_augmented[-1], **inputs
            )

        # Reset learned weights for each new question
        self.reset_learned_aggregation_weights()

        # Apply feature averaging hook if specified
        hook_handle = (
            self.register_feature_averaging_hook()
            if (
                self.token_selection_aggregation_method
                == "average_features_early_layer"
            )
            else None
        )

        # Generate response
        # import pdb
        # pdb.set_trace()
        generated_ids = self.model.generate(**inputs, **self.kwargs)

        # Decode only the new tokens, not the entire sequence
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :],
            skip_special_tokens=True,
        )[-1]

        # Clean up the hook if it was registered
        if hook_handle is not None:
            hook_handle.remove()

        return generated_text.strip()

    def save_inputs_grid_prompts(
        self,
        message,
        formatted_messages,
        images_augmented,
        applied_transforms,
        dataset=None,
    ):
        save_dir_base = f"benchmark_results/n_samples_1000/exp_83_newdatasets_finalized_VISUALSAMPLES/saved_inputs/{str(dataset)}/"
        os.makedirs(save_dir_base, exist_ok=True)

        # Convert PIL images to tensors and create grid
        to_tensor = transforms.ToTensor()
        image_tensors = []
        max_size = 600  # Target size for longest edge

        target_width, target_height = None, None

        for img_list in images_augmented:
            if img_list:  # Check if list is not empty
                img = img_list[0]  # Take first image from each augmentation

                # Calculate scaling ratio based on longest edge
                width, height = img.size
                longest_edge = max(width, height)
                scale_ratio = min(1.0, max_size / longest_edge)

                # Calculate new dimensions while preserving aspect ratio
                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)

                if target_width is None or target_height is None:
                    target_width, target_height = new_width, new_height

                # Resize image with high-quality resampling
                img_resized = img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )

                # Convert to tensor
                img_tensor = to_tensor(img_resized)
                image_tensors.append(img_tensor)

        if image_tensors:
            # Stack tensors into batch
            batch_tensor = torch.stack(image_tensors)
            # Create grid
            img_grid = vutils.make_grid(
                batch_tensor,
                nrow=int(len(image_tensors) ** 0.5),
                padding=2,
                normalize=True,
            )
            # Convert back to PIL and save as JPG
            to_pil = transforms.ToPILImage()
            img_grid_pil = to_pil(img_grid)
            img_grid_save_path = os.path.join(save_dir_base, "image_grid.jpg")
            img_grid_pil.save(img_grid_save_path, "JPEG", quality=95)

        prompts_save_path = os.path.join(save_dir_base, "prompts.txt")
        with open(prompts_save_path, "w") as f:
            for idx, (fmt_msg, transform) in enumerate(
                zip(formatted_messages, applied_transforms)
            ):
                f.write(f"Prompt {idx}:\n{fmt_msg}\n******\n")

    def generate_inner(self, message, dataset=None):
        # print(message, "\n************************")

        formatted_messages, formatted_images = self.build_prompt_cases(message, dataset)
        # print(
        #     "\n***************************************************",
        #     "\n\n\n".join(formatted_messages),
        #     "\n***************************************************",
        # )

        # Convert to list if single image
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        images_augmented, applied_transforms = self.image_augment(images)

        save_visual_samples_flag = os.environ.get(
            "SAVE_VISUAL_SAMPLES", "False"
        ).lower() in ("1", "true", "yes")
        if save_visual_samples_flag:
            self.save_inputs_grid_prompts(
                message,
                formatted_messages,
                images_augmented,
                applied_transforms,
                dataset,
            )

        HANDLE_OUT_OF_MEMORY = getattr(self, "handle_oom", True)
        if not HANDLE_OUT_OF_MEMORY:
            return self.generate_inner_helper(
                message, formatted_messages, images_augmented, dataset
            )
        else:
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    return self.generate_inner_helper(
                        message, formatted_messages, images_augmented, dataset
                    )
                except torch.OutOfMemoryError as e:
                    print(f"Attempt {attempt + 1} failed:", e)
                    images_augmented = images_augmented[
                        : max(1, len(images_augmented) // 2)
                    ]
                    formatted_messages = formatted_messages[
                        : max(1, len(formatted_messages) // 2)
                    ]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all operations complete
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with exception:", e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            else:
                print("All retries failed")
                return ""
