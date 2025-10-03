import hashlib
import inspect
import json
import os
import random
import textwrap

import albumentations as A
import cv2
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import numpy as np
import outlines
import pytesseract
import spacy
import torch
from albumentations.augmentations.transforms import *
from albumentations.core.composition import OneOf
from funcy import print_durations
from openai import OpenAI
from outlines.models.transformers import transformers
from PIL import Image
from pydantic import BaseModel, conlist
from torchvision.transforms import AugMix
from transformers import AutoProcessor


class TextAugment:
    def __init__(
        self,
        n_augmentations: int = 5,
        local_paraphrasing_model=None,
        local_paraphrasing_model_tokenizer=None,
        openai_paraphraser_model: str = "gpt-4o-mini",
        openai_api_key_path: str = "/zhome/88/8/215456/openai_key.txt",
        gpt_paraphraser_strategy="gpt_paraphraser",
        save_or_load=None,
        path_text_aug_to_file="benchmark_results/text_augmentations.json",
        number_of_paraphrased: int = None,
        in_other_words: bool = True,
    ):
        # HYPERPARAMETERS
        # Text related
        self.save_or_load = save_or_load  # TEXT AUGMENTATION CACHING: "save", "load", None. "save" and "load" both save the augmented versions. But, "save" does not use the cached inputs.
        self.path_text_aug_to_file = (
            path_text_aug_to_file  # TEXT AUGMENTATION CACHE FILE PATH
        )
        self.gpt_paraphraser_strategy = getattr(self, gpt_paraphraser_strategy, None)
        # gpt_paraphraser, gpt_paraphraser_beam_like, simple_paraphraser, simple_repeater
        if self.gpt_paraphraser_strategy is None:
            raise ValueError(
                f"Invalid text augmenter strategy: {gpt_paraphraser_strategy}"
            )
        self.number_of_paraphrased = (
            min(number_of_paraphrased, n_augmentations - 1)
            if number_of_paraphrased
            else n_augmentations - 1
        )
        self.local_paraphrasing_model = local_paraphrasing_model
        self.local_paraphrasing_model_tokenizer = local_paraphrasing_model_tokenizer
        self.in_other_words = in_other_words
        ####
        print(
            f"Text augmentation strategy: {gpt_paraphraser_strategy}, number_of_paraphrased: {self.number_of_paraphrased}, in_other_words: {self.in_other_words}"
        )

        self.batch_size = n_augmentations
        self.n_augmentations = (
            n_augmentations - 1
        )  # Number of augmented versions to generate (excluding the original)

        with open(openai_api_key_path, "r") as key_file:
            self.api_key = key_file.read().strip()
        self.client = OpenAI(api_key=self.api_key)
        self.paraphraser_model = openai_paraphraser_model

        # save or load for text augmentations
        self.data_cache = {}
        if self.save_or_load is not None:
            if os.path.exists(self.path_text_aug_to_file):
                with open(self.path_text_aug_to_file, "r") as file:
                    self.data_cache = dict(json.load(file))

        self.smollm_model = None

    def __del__(self):
        self.save_augmentations_cache()

    def save_augmentations_cache(self):
        self.old_data_cache = {}
        if self.save_or_load is not None:
            if os.path.exists(self.path_text_aug_to_file):
                with open(self.path_text_aug_to_file, "r") as file:
                    self.old_data_cache = dict(json.load(file))

        self.data_cache = {**self.old_data_cache, **self.data_cache}

        if self.save_or_load is not None:
            with open(self.path_text_aug_to_file, "w") as file:
                json.dump(self.data_cache, file, indent=4)

    def gpt_paraphraser_beam_like(self, text_prompt: str, n_aug: int) -> list[str]:
        paraphrase_prompt = f"Paraphrase the following text prompt and create one different version of it. Avoid enumerating. Just provide one paraphrased version, do not say something like 'sure! here is your answer'. Your language should be natural. Do not paraphrase the word \"Question:\". Now, paraphrase the following text prompt, everything after this point is the text prompt you should paraphrase:\n\n"

        paraphrased_versions = []
        for i in range(n_aug):
            last_version = (
                paraphrased_versions[-1] if paraphrased_versions else text_prompt
            )
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": paraphrase_prompt + last_version,  # text_prompt,
                    },
                ],
                model=self.paraphraser_model,
            )

            paraphrased_versions.append(
                chat_completion.choices[0].message.content.strip()
            )

        print(text_prompt)
        print(paraphrased_versions)

        return paraphrased_versions

    def gpt_paraphraser_categorical(self, text_prompt: str, n_aug: int) -> list[str]:
        if self.save_or_load == "load":
            loaded = self.mock_gpt_paraphraser(text_prompt, n_aug)
            if loaded:
                return loaded

        # paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs. Between different versions put a <VERSION> separator. Do not alter any <tokens> enclosed within < and > symbols. Additionally, keep the following words and phrases unchanged: User, Assistant, Give a very brief answer, Choices, Answer with a letter, Answer, Question.\n\n\nRemember, you should put '<VERSION>' separators between different versions, this is very important. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt:\n\n"
        # paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs. Text prompt will be a question. You can add some hints relevant to the text prompt in order to make the solution easier. Between different versions put a <VERSION> separator. You should put '<VERSION>' separators between different versions, this is very important. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt, everything after this point is the text prompt you should paraphrase:\n\n"

        ##############
        paraphrase_prompt = textwrap.dedent(f"""
            **System Role:**
            You are a paraphrasing assistant trained to produce controlled and meaningful variations of a sentence. You will generate **exactly 7 paraphrased versions**, each using **one and only one** of the following strategies, applied in the exact order specified below.

            ---

            ### 🔢 **Fixed Strategy Order**

            1. **Change the Tense** — Change the tense of the sentence (e.g., present → past, past → future)
            2. **Replace Verbs with Synonyms** — Substitute the main verbs with accurate synonyms, keeping tense and structure intact
            3. **Switch Voice** — Convert between active and passive voice
            4. **Change Word Order Slightly** — Rearrange the sentence word order while preserving grammaticality and meaning
            5. **Adjust Sentence Structure** — Break long sentences into shorter ones, or combine short ones into a longer one
            6. **Change the Tone** — Shift the tone to be more formal or more informal
            7. **Figurative ↔ Literal Language** — Replace idioms/figurative expressions with literal ones, or vice versa

            ---

            ### 🔧 **Instructions**

            * Apply exactly **one** strategy per version, in the **above order**.
            * Do not use any other paraphrasing techniques.
            * Keep the meaning of the original sentence intact.
            * Separate each version using the token `<VERSION>`, written exactly like that.
            * Do not number, label, or explain the outputs. Just output the paraphrased versions separated by `<VERSION>`.

            ---

            ### 📚 **Few-Shot Example**

            **Input Sentence:**

            ```
            I finished the report before the deadline.
            ```

            **Output:**

            ```
            I will finish the report before the deadline. <VERSION>
            I completed the report before the deadline. <VERSION>
            The report was finished by me before the deadline. <VERSION>
            Before the deadline, I finished the report. <VERSION>
            I finished the report. It was ahead of the deadline. <VERSION>
            The report was done ahead of schedule, sir. <VERSION>
            I wrapped it up before time ran out. <VERSION>
            ```

            ---

            ### ✏️ Now Paraphrase the Following Sentence

            **Input Sentence:**

            \n\n
        """)

        # paraphrase_prompt = f"""You are an expert paraphraser. Your task is to generate {n_aug} paraphrased versions of the given text prompt. Paraphrase the following text prompt and generate {n_aug} variations. The paraphrasing should not change the meaning. Do not try to shorten it. Preserve the meaning, even the details are very important. Just change the wording. Also, add a role related to the question as a prefix.

        # Output format:
        # - Separate each version with "<VERSION>" (exactly like this, no spaces).
        # - Do not number or introduce the outputs.
        # - Only return the paraphrased versions.
        # - Paraphrased versions should start with a role related to the question.

        # Example for 3 paraphrases:
        # _Input:_ "Describe a sunset over the ocean in a poetic way."
        # _Output:_
        # <VERSION>
        # Suppose you are a poet. Paint a poetic picture of an ocean sunset.
        # <VERSION>
        # Imagine you are a painter. Write a lyrical description of the sun setting over the sea.
        # <VERSION>
        # Assume you are a storyteller. Create a poetic depiction of the sun sinking into the ocean.

        # Now, paraphrase the following text prompt:

        # """

        # Generate N augmented text prompts by randomly altering the original text prompt
        chat_completion = self.client.chat.completions.create(
            messages=[
                # {
                #     "role": "system",
                #     "content": paraphrase_prompt,
                # },
                {
                    "role": "user",
                    "content": paraphrase_prompt + text_prompt,
                },
            ],
            model=self.paraphraser_model,
        )

        paraphrased_versions = [
            line
            for line in chat_completion.choices[0].message.content.split("<VERSION>")
            if line.strip()
        ]

        self.data_cache[text_prompt] = paraphrased_versions

        return paraphrased_versions

    def smollm_paraphraser(
        self, text_prompt: str, n_aug: int, in_other_words: bool = True
    ) -> list[str]:
        if self.save_or_load == "load":
            loaded = self.mock_gpt_paraphraser(text_prompt, n_aug)
            if loaded:
                return loaded
        ##############

        # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n" * 3)

        if self.smollm_model is None:
            self.smollm_model = transformers(
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                },  # , "device_map": "auto"},
            )
            self.smollm_model_tokenizer = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            )

            class Paraphrases(BaseModel):
                paraphrases: conlist(str, max_length=n_aug, min_length=n_aug)

            self.structured_generator = outlines.generate.json(
                self.smollm_model, Paraphrases, sampler=outlines.samplers.greedy()
            )

            self.nlp = spacy.load("en_core_web_sm")

            self.save_counter = 0

        def split_sentences(text: str) -> list[str]:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]

        def paraphrase_one_sentence(input_text, n_aug=8):
            try:

                class Paraphrases(BaseModel):
                    paraphrases: conlist(str, max_length=n_aug, min_length=n_aug)

                structured_generator = outlines.generate.json(
                    self.smollm_model, Paraphrases, sampler=outlines.samplers.greedy()
                )

                prompt = textwrap.dedent(f"""
                    You are an expert paraphraser.

                    Your task is to paraphrase input text without changing its meaning. Keep the details and core content. Generate {n_aug} paraphrased versions.

                    Return your output as a JSON object with the key "paraphrases", mapped to a list of {n_aug} unique paraphrased versions.

                    Now, paraphrase the following text:
                    \"\"\"{input_text}\"\"\"
                """)

                messages = [{"role": "user", "content": prompt}]

                prompt_chat_template = self.smollm_model_tokenizer.apply_chat_template(
                    messages, tokenize=False
                )

                output = structured_generator(prompt_chat_template, max_tokens=2048)

                print(
                    "****\nParaphrasing input text - smollm para:",
                    input_text,
                    "\nParaphrased output - smollm para:",
                    output,
                    "\n****",
                )

                result = [
                    para
                    for para in list(set(output.paraphrases))
                    if "Paraphrase" not in para
                ]

                return result if result else [input_text]
            except Exception as e:
                # Fallback to simple_paraphraser if JSON decode error or any other error occurs
                print(
                    f"Error in paraphrase_one_sentence: {e}. Using fallback simple_paraphraser."
                )
                return self.simple_paraphraser(input_text, n_aug, in_other_words=False)

        try:
            example_prompt_splitted_into_sentences = split_sentences(text_prompt)

            paraphrase_system_prompt_prefix = textwrap.dedent(f"""
                You are an expert paraphraser.

                Your task is to paraphrase input text without changing its meaning. Keep the details and core content. Generate {n_aug} paraphrased versions.

                Return your output as a JSON object with the key "paraphrases", mapped to a list of {n_aug} unique paraphrased versions.

                Now, paraphrase the following text:
            """)

            paraphraser_model_prompts = [
                [
                    {
                        "role": "user",
                        "content": paraphrase_system_prompt_prefix + f"\n{sentence}",
                    }
                ]
                for sentence in example_prompt_splitted_into_sentences
            ]
            paraphraser_model_prompts = [
                self.smollm_model_tokenizer.apply_chat_template(prompt, tokenize=False)
                for prompt in paraphraser_model_prompts
            ]
            print("DEBUG POINT 0")
            print(paraphraser_model_prompts)

            batch_output = self.structured_generator(
                paraphraser_model_prompts, max_tokens=2048
            )

            print("DEBUG POINT 1")
            print(batch_output)

            paraphrased_sentences = [
                [
                    para
                    for para in list(set(output.paraphrases))
                    if "Paraphrase" not in para
                ]
                if output.paraphrases
                else [example_prompt_splitted_into_sentences[idx]]
                for idx, output in enumerate(batch_output)
            ]
            print("DEBUG POINT 2")

            # Use random sampling instead of cartesian product to prevent memory explosion
            result = []
            for _ in range(n_aug):  # Limit to prevent memory issues
                combination = []
                for sentence_paraphrases in paraphrased_sentences:
                    if sentence_paraphrases:
                        combination.append(random.choice(sentence_paraphrases))
                    else:
                        combination.append("")
                result.append(" ".join(combination).strip())

            print("DEBUG POINT 3")

            if len(result) < n_aug:
                paraphrased_versions = result + [text_prompt] * (n_aug - len(result))
            else:
                paraphrased_versions = result[:n_aug]
        except Exception as e:
            # simple paraphraser fallback
            print(
                f"Error in batch paraphrasing with smollm: {e}. Using fallback simple_paraphraser."
            )
            paraphrased_versions = self.simple_paraphraser(
                text_prompt, n_aug, in_other_words=False
            )

        print(paraphrased_versions)

        # paraphrased_sentences = [
        #     paraphrase_one_sentence(sentence, n_aug)
        #     for sentence in example_prompt_splitted_into_sentences
        # ]

        if in_other_words:
            paraphrased_versions = [
                version + " In other words, " + text_prompt
                for version in paraphrased_versions
                if isinstance(version, str) and version.strip()
            ]

        # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n" * 3)

        ##############
        self.data_cache[text_prompt] = paraphrased_versions
        self.save_counter += 1
        if self.save_counter % 20 == 19:
            self.save_augmentations_cache()

        return paraphrased_versions

    def gpt_paraphraser(
        self, text_prompt: str, n_aug: int, in_other_words: bool = True
    ) -> list[str]:
        if self.save_or_load == "load":
            loaded = self.mock_gpt_paraphraser(text_prompt, n_aug)
            if loaded:
                paraphrased_versions = loaded
            else:
                # paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs. Between different versions put a <VERSION> separator. Do not alter any <tokens> enclosed within < and > symbols. Additionally, keep the following words and phrases unchanged: User, Assistant, Give a very brief answer, Choices, Answer with a letter, Answer, Question.\n\n\nRemember, you should put '<VERSION>' separators between different versions, this is very important. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt:\n\n"
                # paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs. Text prompt will be a question. You can add some hints relevant to the text prompt in order to make the solution easier. Between different versions put a <VERSION> separator. You should put '<VERSION>' separators between different versions, this is very important. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt, everything after this point is the text prompt you should paraphrase:\n\n"

                ##############
                paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs, do not say something like 'sure! here is your answer'. Between different versions put a <VERSION> separator. You should put '<VERSION>' separators between different versions, this is very important. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt, everything after this point is the text prompt you should paraphrase:\n\n"

                # paraphrase_prompt = f"""You are an expert paraphraser. Your task is to generate {n_aug} paraphrased versions of the given text prompt. Paraphrase the following text prompt and generate {n_aug} variations. The paraphrasing should not change the meaning. Do not try to shorten it. Preserve the meaning, even the details are very important. Just change the wording. Also, add a role related to the question as a prefix.

                # Output format:
                # - Separate each version with "<VERSION>" (exactly like this, no spaces).
                # - Do not number or introduce the outputs.
                # - Only return the paraphrased versions.
                # - Paraphrased versions should start with a role related to the question.

                # Example for 3 paraphrases:
                # _Input:_ "Describe a sunset over the ocean in a poetic way."
                # _Output:_
                # <VERSION>
                # Suppose you are a poet. Paint a poetic picture of an ocean sunset.
                # <VERSION>
                # Imagine you are a painter. Write a lyrical description of the sun setting over the sea.
                # <VERSION>
                # Assume you are a storyteller. Create a poetic depiction of the sun sinking into the ocean.

                # Now, paraphrase the following text prompt:

                # """

                # Generate N augmented text prompts by randomly altering the original text prompt
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        # {
                        #     "role": "system",
                        #     "content": paraphrase_prompt,
                        # },
                        {
                            "role": "user",
                            "content": paraphrase_prompt + text_prompt,
                        },
                    ],
                    model=self.paraphraser_model,
                )

                paraphrased_versions = [
                    line
                    for line in chat_completion.choices[0].message.content.split(
                        "<VERSION>"
                    )
                    if line.strip()
                ]

                self.data_cache[text_prompt] = paraphrased_versions

        if in_other_words:
            paraphrased_versions = [
                version.strip() + " In other words, " + text_prompt
                for version in paraphrased_versions
                if isinstance(version, str) and version.strip()
            ]

        return paraphrased_versions

    def self_paraphraser(self, text_prompt: str, n_aug: int) -> list[str]:
        if self.save_or_load == "load":
            loaded = self.mock_gpt_paraphraser(text_prompt, n_aug)
            if loaded:
                return loaded

        # paraphrase_prompt = f"Paraphrase the following text prompt and create {n_aug} different versions of it. Avoid enumerating. Just provide {n_aug} different outputs, do not say something like 'sure! here is your answer'. Between different versions put a <VERSION> separator. You should put '<VERSION>' separators between different versions, this is very important. No numbering. The separator is '<VERSION>' exactly like this without any space inside the token. Follow these instructions. Do not make any mistake. Now, paraphrase the following text prompt, everything after this point is the text prompt you should paraphrase:\n\n"
        paraphrase_prompt = textwrap.dedent(f"""
        You are an expert paraphraser. Your task is to generate {n_aug} paraphrased versions of the given text prompt. Paraphrase the following text prompt and generate {n_aug} variations. The paraphrasing should not change the meaning. Do not try to shorten it. Preserve the meaning, even the details are very important. Just change the wording.

        Output format:
        - Separate each version with "<VERSION>" (exactly like this, no spaces).
        - Do not number or introduce the outputs.
        - Only return the paraphrased versions.

        Example for 3 paraphrases:
        _Input:_ "Describe a sunset over the ocean in a poetic way."
        _Output:_
        <VERSION>
        Paint a poetic picture of an ocean sunset.
        <VERSION>
        Write a lyrical description of the sun setting over the sea.
        <VERSION>
        Create a poetic depiction of the sun sinking into the ocean.

        Now, paraphrase the following text prompt:

        {text_prompt}
        """)
        # paraphrase_prompt = f"""You are an expert paraphraser. Your task is to generate {n_aug} paraphrased versions of the given text prompt. Paraphrase the following text prompt and generate {n_aug} variations. The paraphrasing should not change the meaning. Do not try to shorten it. Preserve the meaning, even the details are very important. Just change the wording. Also, add a role related to the question as a prefix.

        # Output format:
        # - Separate each version with "<VERSION>" (exactly like this, no spaces).
        # - Do not number or introduce the outputs.
        # - Only return the paraphrased versions.
        # - Paraphrased versions should start with a role related to the question.

        # Example for 3 paraphrases:
        # _Input:_ "Describe a sunset over the ocean in a poetic way."
        # _Output:_
        # <VERSION>
        # Suppose you are a poet. Paint a poetic picture of an ocean sunset.
        # <VERSION>
        # Imagine you are a painter. Write a lyrical description of the sun setting over the sea.
        # <VERSION>
        # Assume you are a storyteller. Create a poetic depiction of the sun sinking into the ocean.

        # Now, paraphrase the following text prompt:

        # {text_prompt}
        # """
        # TODO: grouping, different very specific instructions, change the tense, change the adjectives, change the verb etc

        print("******************")
        print(text_prompt)
        print("******************")

        with torch.no_grad():
            old_token_selection_aggregation_method = self.local_paraphrasing_model.language_model.token_selection_aggregation_method
            self.local_paraphrasing_model.language_model.token_selection_aggregation_method = "original"

            question = paraphrase_prompt
            response, history = self.local_paraphrasing_model.chat(
                self.local_paraphrasing_model_tokenizer,
                None,
                question,
                dict(do_sample=True, max_new_tokens=2048, top_p=None),
                history=None,
                return_history=True,
            )
            self.local_paraphrasing_model.language_model.token_selection_aggregation_method = old_token_selection_aggregation_method

            # print(f"User: {question}\nAssistant: {response}")

        # response = "<VERSION> ".join([text_prompt] * n_aug)

        # paraphrased_versions = [
        #     line.strip() for line in response.split("<VERSION>") if line.strip()
        # ][:n_aug]
        paraphrased_versions = [
            line.strip() + " In other words, " + text_prompt
            for line in response.split("<VERSION>")
            if line.strip()
        ][:n_aug]
        print(paraphrased_versions)
        print("####")

        self.data_cache[text_prompt] = paraphrased_versions

        return paraphrased_versions

    def mock_gpt_paraphraser(self, text_prompt: str, n_aug: int) -> list[str]:
        if text_prompt in self.data_cache:
            result = self.data_cache[text_prompt]

            if len(result) >= n_aug:
                return result

        return False

    def simple_paraphraser(
        self, text_prompt: str, n_aug: int, in_other_words: bool = True
    ) -> list[str]:
        # print("***************")
        # print(text_prompt)
        self.nlpaug = [
            # nas.ContextualWordEmbsForSentenceAug(device="cuda"),
            # naw.SynonymAug(),
            # naw.ContextualWordEmbsAug(device="cuda"),
            nac.KeyboardAug(aug_char_p=0.1, aug_word_p=0.1),
            # naw.BackTranslationAug(device="cuda"),
            naw.SplitAug(),
            naw.RandomWordAug(),
            nas.RandomSentAug(),
            # nas.AbstSummAug(),
        ]

        # self.nlpaug = naf.Sequential(
        #     [
        #         naf.Sometimes(
        #             single_aug,
        #             aug_p=0.3,
        #         )
        #         for single_aug in self.nlpaug
        #     ]
        # )
        self.nlpaug = naf.Sequential(
            [
                single_aug
                for single_aug in np.random.choice(self.nlpaug, 2, replace=False)
            ]
        )

        paraphrased_versions = self.nlpaug.augment(text_prompt, n=n_aug)

        # print("-----------------------------------------")
        # print(paraphrased_versions)
        # print("-----------------------------------------")

        paraphrased_versions = [
            version[0] if isinstance(version, list) else version
            for version in paraphrased_versions
        ]

        if "<image 1>" in text_prompt:
            paraphrased_versions = [
                version.replace("<image>", "<image 1>")
                for version in paraphrased_versions
                if isinstance(version, str) and version.strip()
            ]

        if in_other_words:
            paraphrased_versions = [
                version + " In other words, " + text_prompt
                for version in paraphrased_versions
                if isinstance(version, str) and version.strip()
            ]
        else:
            paraphrased_versions = [
                version
                for version in paraphrased_versions
                if isinstance(version, str) and version.strip()
            ]

        return paraphrased_versions

    def simple_repeater(self, text_prompt: str, n_aug: int) -> list[str]:
        return [text_prompt] * self.n_augmentations  # FOR ONLY IMG AUG

    @print_durations()
    def __call__(self, text_prompt: str) -> list[str]:
        # print("#####")
        # print(text_prompt)
        # print("#####")
        if self.n_augmentations == 0:
            return [text_prompt]

        if (
            "in_other_words"
            in inspect.signature(self.gpt_paraphraser_strategy).parameters
        ):
            print(
                "Using in_other_words in",
                self.gpt_paraphraser_strategy.__name__,
                self.in_other_words,
            )
            paraphrased_versions = self.gpt_paraphraser_strategy(
                text_prompt, self.n_augmentations, in_other_words=self.in_other_words
            )
        else:
            paraphrased_versions = self.gpt_paraphraser_strategy(
                text_prompt, self.n_augmentations
            )

        paraphrased_versions = [version.strip() for version in paraphrased_versions]
        paraphrased_versions = paraphrased_versions[: self.number_of_paraphrased]

        if len(paraphrased_versions) <= self.n_augmentations:
            return paraphrased_versions + [text_prompt] * (
                self.n_augmentations - len(paraphrased_versions) + 1
            )

        return paraphrased_versions[: self.n_augmentations] + [text_prompt]
