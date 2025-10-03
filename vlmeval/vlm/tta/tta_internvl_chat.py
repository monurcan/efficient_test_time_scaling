import contextlib
import copy
import os  # Add import for os
import time
import types
from functools import partial, partialmethod

import numpy as np
import torch
from PIL import Image
from torch.nn import CosineSimilarity
from torchvision import transforms
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer,
)
from transformers.image_utils import load_image as load_image_transformers

from ...dataset import DATASET_MODALITY, DATASET_TYPE
from ...smp import *
from ..base import BaseModel
from ..internvl.internvl_chat import InternVLChat
from ..internvl.utils import (
    build_mcq_cot_prompt,
    build_mpo_prompt,
    build_multi_choice_prompt,
    build_qa_cot_prompt,
    build_transform,
    build_video_prompt,
    dynamic_preprocess,
    load_image,
    mpo_post_processing,
    mpo_prompt_with_final_answer,
    mpo_prompt_without_final_answer,
    reorganize_prompt,
    split_model,
)
from .image_augment import ImageAugment
from .modified_sample_451 import _modified_sample
from .text_augment import TextAugment


class TTAugAdapter_InternVLChat(InternVLChat):
    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        print("================================")
        print(f"Initializing TTAugAdapter_InternVLChat")
        print(f"Adapter Args: {adapter_args}")
        print("================================")

        for key, value in adapter_args.items():
            setattr(self, key, value)

        super().__init__(**model_args)

        self.text_augment = TextAugment(
            n_augmentations=self.number_of_versions,
            local_paraphrasing_model=self.model,
            local_paraphrasing_model_tokenizer=self.tokenizer,
            **text_aug_args,
        )

        self.image_augment = ImageAugment(
            n_augmentations=self.number_of_versions, **image_aug_args
        )

        self.model.language_model.token_selection_aggregation_method = (
            self.token_selection_aggregation_method
        )
        self.model.language_model._sample = types.MethodType(
            _modified_sample, self.model.language_model
        )

        self.model.number_of_versions = self.number_of_versions
        print(f"{self.use_cot=}, {self.use_mpo_prompt=}")
        if self.use_cot:
            os.environ["USE_COT"] = "1"

    def build_prompt(self, line, dataset=None):
        all_versions = []

        # Only change the question, not hints or choices.
        augmented_versions = self.text_augment(line["question"])

        for augmented_version in augmented_versions:
            modified_line = copy.deepcopy(line)
            modified_line["question"] = augmented_version
            single_output = super().build_prompt(modified_line, dataset)
            all_versions.extend(single_output)

        return all_versions

    def generate_v2_helper(self, dataset, pixel_values, num_patches_list, questions):
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ["MMStar", "HallusionBench", "OCRBench"]
        )
        # print(f"Using MPO Prompt: {use_mpo_prompt=}")
        # print(f"{questions=}")
        # print(f"{pixel_values=}")
        # print("*************\n\n\n")

        with (
            torch.no_grad()
            if self.token_selection_aggregation_method != "learned"
            else torch.enable_grad()
        ):
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=self.kwargs,
                # verbose=True,
            )
            response = responses[-1]

        if use_mpo_prompt:
            response = mpo_post_processing(response, dataset)

        return response

    def generate_v2(self, message, dataset=None):
        # print(
        #     "==============================\n\n",
        #     message,
        #     "==============================\n\n",
        # )
        chunk_size = len(message) // self.number_of_versions
        grouped_message = [
            message[i : i + chunk_size] for i in range(0, len(message), chunk_size)
        ]

        num_patches_list, pixel_values_list = [], []
        questions = []

        message = grouped_message[0]
        images_to_augment = []

        image_num = len([x for x in message if x["type"] == "image"])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        image_path = [x["value"] for x in message if x["type"] == "image"]
        for image_idx, file_name in enumerate(image_path):
            upscale_flag = (
                image_idx == 0 and dataset is not None and listinstr(["MMMU"], dataset)
            )
            image = Image.open(file_name).convert("RGB")
            if upscale_flag:
                image = image.resize(
                    (image.width * 2, image.height * 2), Image.BILINEAR
                )

            images_to_augment.append(image)

        images_augmented, applied_transforms = self.image_augment(images_to_augment)
        images_augmented = [item for sublist in images_augmented for item in sublist]

        for image in images_augmented:
            input_size = 448
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            curr_pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
            num_patches_list.append(curr_pixel_values.size(0))
            pixel_values_list.append(curr_pixel_values)

        pixel_values = (
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        )

        for message in grouped_message:
            prompt = reorganize_prompt(message, image_num, dataset=dataset)

            if dataset is not None and DATASET_MODALITY(dataset) == "VIDEO":
                prompt = build_video_prompt(prompt, dataset)

            questions.append(prompt)

        # print("*.*.*.**.*.*.*..*.*")
        # print(len(questions), pixel_values.size(), len(num_patches_list))
        # print("*.*.*.**.*.*.*..*.*")

        HANDLE_OUT_OF_MEMORY = getattr(self, "handle_oom", True)
        if not HANDLE_OUT_OF_MEMORY:
            return self.generate_v2_helper(
                dataset, pixel_values, num_patches_list, questions
            )
        else:
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    return self.generate_v2_helper(
                        dataset, pixel_values, num_patches_list, questions
                    )
                except torch.OutOfMemoryError as e:
                    print(f"Attempt {attempt + 1} failed:", e)
                    pixel_values = pixel_values[: max(1, pixel_values.size(0) // 2)]
                    num_patches_list = num_patches_list[
                        : max(1, len(num_patches_list) // 2)
                    ]
                    questions = questions[: max(1, len(questions) // 2)]

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
