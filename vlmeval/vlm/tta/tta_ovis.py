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
)

from ...dataset import DATASET_MODALITY, DATASET_TYPE
from ...smp import *
from ..ovis.ovis import Ovis2
from .image_augment import ImageAugment
from .modified_sample_451 import _modified_sample
from .text_augment import TextAugment


class TTAugAdapter_Ovis2(Ovis2):
    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        print("================================")
        print(f"Initializing TTAugAdapter_Ovis2")
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
            self.model_unsloth.llm.token_selection_aggregation_method = (
                self.token_selection_aggregation_method
            )

            self.model_unsloth.llm._sample = types.MethodType(
                _modified_sample, self.model_unsloth
            )
        else:
            self.model.llm.token_selection_aggregation_method = (
                self.token_selection_aggregation_method
            )

            self.model.llm._sample = types.MethodType(_modified_sample, self.model.llm)

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

    def build_prompt(self, line, dataset=None):
        all_versions = []

        # Only change the question, not hints or choices.
        augmented_versions = self.text_augment(line["question"])

        for augmented_version in augmented_versions:
            modified_line = copy.deepcopy(line)
            modified_line["question"] = augmented_version

            tgt_path = self.dump_image(modified_line, dataset)

            # print(f"{DATASET_TYPE(dataset)=}")

            if DATASET_TYPE(dataset) == "Y/N":
                prompt = self.build_yorn_prompt(modified_line, dataset)
            elif DATASET_TYPE(dataset) == "MCQ":
                prompt = self.build_multi_choice_prompt(modified_line, dataset)
            elif DATASET_TYPE(dataset) == "VQA":
                prompt = (
                    augmented_version
                    + "\nAnswer the question using a single word or phrase."
                )
            elif DATASET_TYPE(dataset) == "MMERealWorld":
                prompt = augmented_version + (
                    modified_line["multi-choice options"]
                    + "\nAnswer with the option's letter from the given choices directly."
                )
            else:
                prompt = augmented_version

            # print(f"{prompt=}")
            # print(
            #     "\n\n================================\n",
            #     line,
            #     "\n================================\n\n",
            # )

            single_output = [dict(type="text", value=prompt)]
            single_output.extend([dict(type="image", value=s) for s in tgt_path])

            all_versions.extend(single_output)

        return all_versions

    def load_model_with_unsloth(self):
        """Load model using unsloth for efficient training."""
        print("Loading model with Unsloth...")

        model_path = self.model_args.get("model_path", "AIDC-AI/Ovis2-2B")

        # Load model and processor using unsloth with new options
        self.model_unsloth, self.processor = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=True,
            # use_gradient_checkpointing="unsloth",
            use_gradient_checkpointing=False,
            trust_remote_code=True
        )

        print(f"Unsloth model loaded successfully: {type(self.model_unsloth)}")

        # Store initial model state for resetting weights
        self._store_initial_model_state()

        # Setup unsloth adapter immediately
        print("Setting up Unsloth adapter for Ovis2")
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

    def learned_model_generate(self, grouped_message): #formatted_messages, images_augmented):
        """Generate using iterative pseudolabel training with unsloth."""
        print("Starting pseudolabel-based generation...")
        print("!! NOT IMPLEMENTED due to Unsloth incompatibility with Ovis2")
        print(grouped_message)
        
        return ""

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
            "**********************FINAL ANSWER***************************",
        )

        # Return aggregated response (take first if multiple)
        return generated_text if generated_text else ""

        """Reset learned aggregation weights and model weights for each new question."""
        # Reset aggregation-related attributes
        if hasattr(self.model, "aggregation_weights"):
            delattr(self.model, "aggregation_weights")
        if hasattr(self.model, "_tta_optimizer"):
            delattr(self.model, "_tta_optimizer")
        if hasattr(self.model, "_token_step_count"):
            delattr(self.model, "_token_step_count")

    def use_custom_prompt(self, dataset):
        # always call build_prompt() to enable text augmentations!
        return True

    def generate_inner_helper(
        self, batch_input_ids, batch_attention_mask, batch_pixel_values, last_prompt, grouped_message
    ):
        def _extract_answer(text):
            answer_index = text.lower().find("the answer is")
            if answer_index != -1:
                answer_index += len("the answer is")
                answer = text[answer_index:].lstrip(":").strip()
            else:
                answer = text
            return answer
        
        if self.token_selection_aggregation_method == "learned_model":
            response = self.learned_model_generate(grouped_message)
        else:
            # Generate outputs using batch inference
            with torch.inference_mode():
                output_ids = self.model.generate(
                    batch_input_ids,
                    pixel_values=batch_pixel_values,
                    attention_mask=batch_attention_mask,
                    **self.gen_kwargs,
                )

            # Decode the first output (all versions should be similar due to aggregation)
            response = self.text_tokenizer.decode(
                output_ids[-1], skip_special_tokens=True
            ).strip()

        if (
            "conclude with 'the answer is' followed by the final solution."
            in last_prompt
        ):
            response = _extract_answer(response)

        return response

    def prepare_inputs(self, message, dataset=None):
        # build query
        images = [x["value"] for x in message if x["type"] == "image"]
        texts = [x["value"] for x in message if x["type"] == "text"]
        if DATASET_MODALITY(dataset) == "VIDEO":  # video inputs
            chunks = [self.image_placeholder for x in message if x["type"] != "text"]
            chunks += [
                x["value"].strip()
                for x in message
                if x["type"] == "text" and x["value"] != ""
            ]
            query = "\n".join(chunks)
        elif len(images) == 0:  # text-only inputs
            query = "\n".join(texts)
        elif len(images) == 1 and len(texts) == 1:  # single-image inputs
            query = self.image_placeholder + "\n" + texts[0]
        else:  # interleaved inputs
            chunks = [
                x["value"].strip() if x["type"] == "text" else self.image_placeholder
                for x in message
            ]
            query = "\n".join(chunks)

        # preprocess inputs
        if DATASET_MODALITY(dataset) == "VIDEO":
            max_partition = 1
        elif any(
            dataset.startswith(prefix)
            for prefix in (
                "HallusionBench",
                "TextVQA",
                "ChartQA",
                "OCRBench",
                "InfoVQA",
                "DocVQA",
                "MTVQA",
            )
        ):
            max_partition = 12
        else:
            max_partition = 9
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query,
            [image for image in images],
            max_partition=max_partition,
            frame_selector=self.frame_selector,
        )

        # move to self.device
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = [
            pixel_values.to(device=self.device, dtype=self.dtype)
            if pixel_values is not None
            else None
        ]

        return prompt, input_ids, attention_mask, pixel_values, max_partition

    def generate_inner(self, message, dataset=None):
        # print(f"{message=}")

        chunk_size = len(message) // self.number_of_versions
        grouped_message = [
            message[i : i + chunk_size] for i in range(0, len(message), chunk_size)
        ]

        # print(f"{grouped_message=}")

        image_to_augment = [
            Image.open(x["value"]).convert("RGB")
            for x in message
            if x["type"] == "image"
        ][0]
        print(grouped_message[-1])
        # print(grouped_message)
        # print(f"Original image shape: {image_to_augment.size}")
        images_augmented, applied_transforms = self.image_augment(image_to_augment)
        # print(f"Augmented {len(images_augmented)} images")
        # print(f"{len(grouped_message)=}, {len(images_augmented)=}")

        for i, chunk in enumerate(grouped_message):
            # Replace image placeholder with actual augmented image
            for msg in chunk:
                if msg.get("type") == "image":
                    msg["value"] = images_augmented[i][0]
                    break
        # print(f"Prepared {len(grouped_message)} message groups with augmented images")
        # print(grouped_message)

        # Prepare batch inputs
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        last_prompt = ""

        for chunk in grouped_message:
            prompt, input_ids, attention_mask, pixel_values, max_partition = (
                self.prepare_inputs(chunk, dataset)
            )
            # print(f"{prompt=}")
            # print(
            #     f"{input_ids.shape=}, {attention_mask.shape=}, {pixel_values[0].shape=}"
            # )

            batch_input_ids.append(input_ids[0].to(device=self.device))
            batch_attention_mask.append(attention_mask[0].to(device=self.device))
            batch_pixel_values.append(
                pixel_values[0].to(dtype=self.dtype, device=self.device)
            )
            last_prompt = prompt

        # Pad sequences for batch processing
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_input_ids],
            batch_first=True,
            padding_value=0.0,
        ).flip(dims=[1])
        batch_input_ids = batch_input_ids[:, -self.model.config.multimodal_max_length :]

        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_attention_mask],
            batch_first=True,
            padding_value=False,
        ).flip(dims=[1])
        batch_attention_mask = batch_attention_mask[
            :, -self.model.config.multimodal_max_length :
        ]

        ##################
        # print(
        #     f"{len(batch_input_ids)=}, {len(batch_attention_mask)=}, {len(batch_pixel_values)=}"
        # )
        HANDLE_OUT_OF_MEMORY = getattr(self, "handle_oom", True)
        if not HANDLE_OUT_OF_MEMORY:
            return self.generate_inner_helper(
                batch_input_ids, batch_attention_mask, batch_pixel_values, last_prompt, grouped_message
            )

        else:
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    return self.generate_inner_helper(
                        batch_input_ids,
                        batch_attention_mask,
                        batch_pixel_values,
                        last_prompt,
                        grouped_message
                    )

                except torch.OutOfMemoryError as e:
                    print(f"Attempt {attempt + 1} failed:", e)
                    # Reduce batch size or sequence length
                    batch_input_ids = batch_input_ids[len(batch_input_ids) // 2 :]
                    batch_attention_mask = batch_attention_mask[
                        len(batch_attention_mask) // 2 :
                    ]
                    batch_pixel_values = batch_pixel_values[
                        len(batch_pixel_values) // 2 :
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
