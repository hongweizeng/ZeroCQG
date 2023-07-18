# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""CRD3  dataset"""


import json
import os
import random
import numpy
from nltk import word_tokenize, sent_tokenize

import datasets


_CITATION = """
@misc{campos2020doqa,
    title={DoQA -- Accessing Domain-Specific FAQs via Conversational QA},
    author={Jon Ander Campos and Arantxa Otegi and Aitor Soroa and Jan Deriu and Mark Cieliebak and Eneko Agirre},
    year={2020},
    eprint={2005.01328},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
 """

_DESCRIPTION = """
DoQA is a dataset for accessing Domain Specific FAQs via conversational QA that contains 2,437 information-seeking question/answer dialogues
(10,917 questions in total) on three different domains: cooking, travel and movies. Note that we include in the generic concept of FAQs also
Community Question Answering sites, as well as corporate information in intranets which is maintained in textual form similar to FAQs, often
referred to as internal “knowledge bases”.
These dialogues are created by crowd workers that play the following two roles: the user who asks questions about a given topic posted in Stack
Exchange (https://stackexchange.com/), and the domain expert who replies to the questions by selecting a short span of text from the long textual
reply in the original post. The expert can rephrase the selected span, in order to make it look more natural. The dataset covers unanswerable
questions and some relevant dialogue acts.
DoQA enables the development and evaluation of conversational QA systems that help users access the knowledge buried in domain specific FAQs.
"""

_URL = "http://ixa2.si.ehu.es/convai/doqa-v2.1.zip"


class Doqa(datasets.GeneratorBasedBuilder):

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
            homepage="http://ixa.eus/node/12931",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "filepath": os.path.join(path, "doqa-v2.1", "doqa_dataset", "doqa-cooking-train-v2.1.json"), "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "filepath": os.path.join(path, "doqa-v2.1", "doqa_dataset", "doqa-cooking-dev-v2.1.json"), "split": "validation"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "filepath": {
                        "cooking": os.path.join(path, "doqa-v2.1", "doqa_dataset", "doqa-cooking-test-v2.1.json"),
                        "movies": os.path.join(path, "doqa-v2.1", "doqa_dataset", "doqa-movies-test-v2.1.json"),
                        "travel": os.path.join(path, "doqa-v2.1", "doqa_dataset", "doqa-travel-test-v2.1.json")
                    }, "split": "test"}
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        random.seed(42)

        examples = []
        count = 1

        if isinstance(filepath, dict):
            filepath = filepath.values()
        elif isinstance(filepath, str):
            filepath = [filepath]
        else:
            raise NotImplementedError(f"filepath = {filepath}, instance type = {type(filepath)}")

        for fp in filepath:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
                for row in data["data"]:
                    title = row["title"]
                    background = row["background"]
                    paragraphs = row["paragraphs"]
                    for p in paragraphs:
                        context_text = p["context"]

                        questions, answers = [], []

                        qas = p["qas"]
                        for qa in qas:
                            question_text = qa["question"]
                            answer_text = qa["answers"][0]['text']
                            orig_answer_text = qa["orig_answer"]["text"]

                            questions.append(question_text)
                            answers.append(answer_text)

                        example = {
                            "context": context_text,
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
