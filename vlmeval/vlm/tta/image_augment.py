import hashlib
import json
import os

import albumentations as A
import cv2
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import numpy as np
import pytesseract
import torch
from albumentations.augmentations.transforms import *
from albumentations.core.composition import OneOf
from funcy import print_durations
from openai import OpenAI
from PIL import Image
from torchvision.transforms import AugMix


class ImageAugment:
    def __init__(
        self,
        n_augmentations: int = 5,
        aug_strength="augmix",
        save_or_load_generativeimg=None,
        strength_gen_aug=0.25,
        path_generativeimg_to_directory=None,
    ):
        # HYPERPARAMETERS
        # Image related
        self.aug_strength = aug_strength  # IMAGE_AUGMENTATION SETTING: high, medium, low, no, augmix, generative
        self.save_or_load_generativeimg = save_or_load_generativeimg  # GENERATIVE IMG AUG CACHING: "save", "load", None. "save" and "load" both save the augmented versions. But, "save" does not use the cached inputs.
        self.strength_gen_aug = strength_gen_aug
        self.path_generativeimg_to_directory = (
            f"benchmark_results/generative_img_augmentations{self.strength_gen_aug}_final_exps_/"
            if path_generativeimg_to_directory is None
            else path_generativeimg_to_directory
        )  # GENERATIVE IMG aUG CACHE FOLDER PATH
        ####

        self.batch_size = n_augmentations
        self.n_augmentations = (
            n_augmentations - 1
        )  # Number of augmented versions to generate (excluding the original)

        # self.default_high_aug_list = "augmix"
        self.default_high_aug_list = [
            A.RandomBrightnessContrast(p=0.6),
            A.SafeRotate(limit=20, p=0.6, border_mode=cv2.BORDER_CONSTANT, fill=144),
            A.GaussianBlur(blur_limit=(3, 7), p=0.6),
            A.CLAHE(p=0.5),
            A.RandomGamma(p=0.6),
            A.HueSaturationValue(p=0.6),
            A.RandomScale(scale_limit=0.1, p=0.6),
            A.RGBShift(p=0.6),
            A.MedianBlur(blur_limit=3, p=0.6),
            A.ImageCompression(quality_range=(85, 95), p=0.45),
            A.Sharpen(p=0.6),
            A.PlanckianJitter(),
            A.RandomFog(alpha_coef=0.15),
            A.RandomToneCurve(),
            A.Emboss(),
            A.GridDistortion(),
            A.Perspective(scale=0.05, fit_output=True),
            #
            A.GridDropout(ratio=0.25, random_offset=True, fill=144, p=0.66),
            A.CoarseDropout(fill=144, p=0.7),
        ]

        # Define possible transformations
        if self.aug_strength == "high":
            self.possible_image_processors = self.default_high_aug_list
        elif self.aug_strength == "medium":
            self.possible_image_processors = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.SafeRotate(
                    limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=144
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.CLAHE(clip_limit=3.0, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.5
                ),
                A.RandomScale(scale_limit=0.08, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.ImageCompression(quality_range=(85, 95), p=0.35),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.6, 1.0), p=0.5),
                A.PlanckianJitter(p=0.5),
                A.RandomFog(alpha_coef=0.1, p=0.3),
                A.RandomToneCurve(scale=0.2, p=0.5),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 0.7), p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),
                A.Perspective(scale=0.03, fit_output=True, p=0.5),
                #
                A.GridDropout(ratio=0.25, random_offset=True, fill=144, p=0.6),
                A.CoarseDropout(fill=144, p=0.5),
            ]
        elif self.aug_strength == "low":
            self.possible_image_processors = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.SafeRotate(
                    limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, fill=144
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3
                ),
                A.RandomScale(scale_limit=0.05, p=0.3),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.ImageCompression(quality_range=(85, 95), p=0.25),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.3),
                A.PlanckianJitter(p=0.3),
                A.RandomFog(alpha_coef=0.05, p=0.2),
                A.RandomToneCurve(scale=0.1, p=0.3),
                A.Emboss(alpha=(0.1, 0.3), strength=(0.3, 0.5), p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                A.Perspective(scale=0.02, fit_output=True, p=0.3),
            ]
        elif self.aug_strength == "no":
            self.possible_image_processors = [
                A.Lambda(image=lambda x, **kwargs: x),
                A.Lambda(image=lambda x, **kwargs: x),
                A.Lambda(image=lambda x, **kwargs: x),
            ]
        elif self.aug_strength == "augmix":
            self.possible_image_processors = "augmix"
        elif self.aug_strength == "generative":
            self.possible_image_processors = "generative"
        elif self.aug_strength == "generative+augmix":
            self.possible_image_processors = "generative+augmix"
        elif self.aug_strength == "fixed":
            self.fixed_transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1
                ),
                A.SafeRotate(limit=15, p=1, border_mode=cv2.BORDER_CONSTANT, fill=144),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.CLAHE(clip_limit=3.0, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=1
                ),
                A.RandomScale(scale_limit=0.08, p=1),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.ImageCompression(quality_range=(85, 95), p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.6, 1.0), p=1),
                A.PlanckianJitter(p=1),
                A.RandomFog(alpha_coef=0.1, p=1),
                A.RandomToneCurve(scale=0.2, p=1),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 0.7), p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1),
                A.Perspective(scale=0.03, fit_output=True, p=1),
                A.GridDropout(ratio=0.25, random_offset=True, fill=144, p=1),
                A.CoarseDropout(fill=144, p=1),
            ]
            self.possible_image_processors = "fixed"
        else:
            raise ValueError("Invalid augmentation strength")
        self.N_image_transformations = 3  # One transformation applied to the image = Composition of N_image_transformations transformations

        # save or load for generative image augmentations
        if "generative" in self.possible_image_processors:
            from diffusers import AutoPipelineForImage2Image

            self.gen_pipeline = AutoPipelineForImage2Image.from_pretrained(
                "black-forest-labs/FLUX.1-dev",  # "stabilityai/stable-diffusion-3.5-large", #torch_dtype=torch.bfloat16, use_safetensors=True
                torch_dtype=torch.bfloat16,
            )
            self.gen_pipeline.enable_model_cpu_offload()

            if not os.path.exists(self.path_generativeimg_to_directory):
                os.makedirs(self.path_generativeimg_to_directory)

    def hash_raw_image(self, image, hash_func=hashlib.sha256):
        img_bytes = image.tobytes()
        return hash_func(img_bytes).hexdigest()

    def contains_text(self, image):
        return bool(pytesseract.image_to_string(image).strip())

    def augment_images_generative(self, init_image):
        is_init_image_too_big = init_image.height >= 1000 or init_image.width >= 1000
        is_init_image_too_small = init_image.height <= 20 or init_image.width <= 20

        if (
            self.contains_text(init_image)
            or is_init_image_too_big
            or is_init_image_too_small
        ):
            return None

        hash_img = self.hash_raw_image(init_image)
        if self.save_or_load_generativeimg == "load":
            if hash_img in os.listdir(self.path_generativeimg_to_directory):
                try:
                    variations = [
                        Image.open(
                            os.path.join(
                                self.path_generativeimg_to_directory,
                                hash_img,
                                f"{i}.png",
                            )
                        )
                        for i in range(self.n_augmentations)
                    ]
                    return variations
                except:
                    print("Error loading generative image augmentations, regenerating")

        variations = self.gen_pipeline(
            ["realistic image"] * self.n_augmentations,
            image=init_image,
            height=init_image.height,
            width=init_image.width,
            guidance_scale=3.0,
            strength=self.strength_gen_aug,  # 0.25,0.4,0.5,
        ).images

        if self.save_or_load_generativeimg is not None:
            save_path = os.path.join(self.path_generativeimg_to_directory, hash_img)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i, variation in enumerate(variations):
                variation.save(os.path.join(save_path, f"{i}.png"))

        return variations

    @print_durations()
    def __call__(
        self, input_images: Image.Image | list[Image.Image]
    ) -> tuple[list[list[Image.Image]], list[str]]:
        if isinstance(input_images, Image.Image):
            input_images = [input_images]

        augmented_images = [[] for _ in range(self.n_augmentations)]
        applied_transforms = []

        # Convert PIL Image to NumPy array (albumentations works on NumPy)
        input_nps = []
        for input_image in input_images:
            input_nps.append(np.array(input_image))

        for input_np in input_nps:
            if "generative" in self.possible_image_processors:
                generative_outputs = self.augment_images_generative(
                    Image.fromarray(input_np)
                )
                if generative_outputs is not None:
                    for i, generative_output in enumerate(generative_outputs):
                        augmented_images[i].append(
                            AugMix()(generative_output)
                            if self.possible_image_processors == "generative+augmix"
                            else generative_output
                        )
                    applied_transforms = [
                        [self.possible_image_processors]
                    ] * self.n_augmentations
                    continue
                # else no_aug: [init_image] * self.n_augmentations

            if self.aug_strength == "fixed":
                L = len(self.fixed_transforms)
                for i in range(self.n_augmentations):
                    if i < L:
                        # Use the single fixed transformation.
                        transform = self.fixed_transforms[i]
                        result = transform(image=input_np)
                        applied_trans = [str(transform)]
                    else:
                        # Compose two different transforms from the fixed list.
                        first = self.fixed_transforms[i % L]
                        second = self.fixed_transforms[(i + 1) % L]
                        composed_transform = A.Compose([first, second])
                        result = composed_transform(image=input_np)
                        applied_trans = [str(first), str(second)]
                    augmented_image = Image.fromarray(result["image"])
                    augmented_images[i].append(augmented_image)
                    applied_transforms.append(applied_trans)
            else:
                for i in range(self.n_augmentations):
                    if (
                        self.possible_image_processors == "augmix"
                        or (
                            self.default_high_aug_list == "augmix"
                            and self.possible_image_processors == "generative"
                        )
                        or self.possible_image_processors == "generative+augmix"
                    ):
                        selected_transforms = ["augmix"]
                        augmented_image = AugMix()(Image.fromarray(input_np))
                    else:
                        possible_image_processors = (
                            self.possible_image_processors
                            if isinstance(self.possible_image_processors, list)
                            else self.default_high_aug_list
                        )

                        # Randomly select N transformations from the list
                        selected_transforms = np.random.choice(
                            possible_image_processors,
                            self.N_image_transformations,
                            replace=False,
                        ).tolist()
                        # Create a composed transformation pipeline by applying the selected transforms sequentially
                        composed_transform = A.Compose(selected_transforms)

                        # Apply the transformations to each input image
                        augmented = composed_transform(image=input_np)["image"]
                        augmented_image = Image.fromarray(augmented)

                    augmented_images[i].append(augmented_image)
                    applied_transforms.append(
                        [str(transform) for transform in selected_transforms]
                    )

        # Append the original input image to the augmented images
        augmented_images.append(input_images)
        applied_transforms.append([])

        return augmented_images, applied_transforms
