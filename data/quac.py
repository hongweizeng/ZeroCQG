# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QUAC (Question Answering in Context)."""


import json
import random
import numpy
from nltk import word_tokenize, sent_tokenize

import datasets

import torch
from transformers import BertTokenizer, BertForNextSentencePrediction


_CITATION = """\
@inproceedings{choi-etal-2018-quac,
title = "QUAC: Question answering in context",
abstract = "We present QuAC, a dataset for Question Answering in Context that contains 14K information-seeking QA dialogs (100K questions in total). The dialogs involve two crowd workers: (1) a student who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a teacher who answers the questions by providing short excerpts from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context, as we show in a detailed qualitative evaluation. We also report results for a number of reference models, including a recently state-of-the-art reading comprehension architecture extended to model dialog context. Our best model underperforms humans by 20 F1, suggesting that there is significant room for future work on this data. Dataset, baseline, and leaderboard available at http://quac.ai.",
author = "Eunsol Choi and He He and Mohit Iyyer and Mark Yatskar and Yih, {Wen Tau} and Yejin Choi and Percy Liang and Luke Zettlemoyer",
year = "2018",
language = "English (US)",
series = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018",
publisher = "Association for Computational Linguistics",
pages = "2174--2184",
editor = "Ellen Riloff and David Chiang and Julia Hockenmaier and Jun'ichi Tsujii",
booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018",
note = "2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018 ; Conference date: 31-10-2018 Through 04-11-2018",
}
"""

_DESCRIPTION = """\
Question Answering in Context is a dataset for modeling, understanding,
and participating in information seeking dialog. Data instances consist
of an interactive dialog between two crowd workers: (1) a student who
poses a sequence of freeform questions to learn as much as possible
about a hidden Wikipedia text, and (2) a teacher who answers the questions
by providing short excerpts (spans) from the text. QuAC introduces
challenges not found in existing machine comprehension datasets: its
questions are often more open-ended, unanswerable, or only meaningful
within the dialog context.
"""

_HOMEPAGE = "https://quac.ai/"

_LICENSE = "MIT"

_URLs = {
    "train": "https://s3.amazonaws.com/my89public/quac/train_v0.2.json",
    "validation": "https://s3.amazonaws.com/my89public/quac/val_v0.2.json",
}


class Quac(datasets.GeneratorBasedBuilder):
    """QuAC (Question Answering in Context)."""

    VERSION = datasets.Version("1.1.0")


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "history_questions": datasets.features.Sequence(datasets.Value("string")),
                    "history_answers": datasets.features.Sequence(datasets.Value("string")),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"], "split": "train"
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "filepath": data_dir["validation"], "split": "validation"
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["validation"], "split": "test"
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        device = torch.device(0)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        bert_model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        bert_model = bert_model.to(device)

        examples = []

        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            data = squad["data"]

            for section in data:
                wiki_page_title = section.get("title", "").strip()
                background = section.get("background", "").strip()
                section_title = section.get("section_title", "").strip()

                for dialogue in section["paragraphs"]:
                    context = dialogue["context"].strip()
                    dialogue_id = dialogue["id"]

                    # context = context[:-12]
                    # context = " ".join(word_tokenize(context.strip()))

                    questions = []
                    answers = []

                    for turn in dialogue["qas"]:

                        question_text = turn["question"]
                        answer_text = turn["answers"][0]["text"].strip()
                        # answer_text = turn["orig_answer"]["text"].strip()


                        questions.append(question_text)
                        answers.append(answer_text)


                    example = {
                        "context": context,
                        "questions": questions,
                        "answers": answers,
                    }
                    examples.append(example)



        # 3. Main task: the conversational question generation with context, history qa pairs, and answer.
        main_examples = []
        for example in examples:
            context_text = example['context']
            questions = example['questions']
            answers = example['answers']

            context_text = " ".join(word_tokenize(context_text.strip()))

            temp_questions, temp_answers = [], []
            num_turns = len(questions)

            for turn_idx in range(num_turns):
                question_text = questions[turn_idx]
                answer_text = answers[turn_idx]

                # Remove the general answer which contains too little information for QG.
                if answer_text == "CANNOTANSWER" or answer_text.lower() == "no" or answer_text.lower() == "yes" or answer_text.lower() == "unknown":
                    continue

                question_text = " ".join(word_tokenize(question_text.strip()))
                answer_text = " ".join(word_tokenize(answer_text.strip()))

                temp_questions.append(question_text)
                temp_answers.append(answer_text)


                # if len(temp_questions) > 0:
                if len(temp_questions) > 1:

                    history_questions, question_text = temp_questions[:-1], temp_questions[-1]
                    history_answers, answer_text = temp_answers[:-1], temp_answers[-1]

                    new_example = {
                        "context": context_text,
                        "history_questions": history_questions,
                        "history_answers": history_answers,
                        "question": question_text,
                        "answer": answer_text,
                    }
                    main_examples.append(new_example)

        # 4. yield
        for idx, ex in enumerate(main_examples):
            yield idx, ex

