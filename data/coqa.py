"""TODO(coqa): Add a description here."""


import re
import json
import random
from nltk import word_tokenize, sent_tokenize
import numpy
import logging

import datasets
# logger = datasets.logging.get_logger(__name__)

import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

logger = logging.getLogger(__name__)


# TODO(coqa): BibTeX citation
_CITATION = """\
@InProceedings{SivaAndAl:Coca,
       author = {Siva, Reddy and Danqi, Chen and  Christopher D., Manning},
        title = {WikiQA: A Challenge Dataset for Open-Domain Question Answering},
      journal = { arXiv},
         year = {2018},
}
"""

# TODO(coqa):
_DESCRIPTION = """\
CoQA: A Conversational Question Answering Challenge
"""

_TRAIN_DATA_URL = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json"
_DEV_DATA_URL = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"


class Coqg(datasets.GeneratorBasedBuilder):
    """TODO(coqa): Short description of my dataset."""

    # TODO(coqa): Set up version.
    VERSION = datasets.Version("0.0.1")

    def _info(self):
        # TODO(coqa): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "history_questions": datasets.features.Sequence(datasets.Value("string")),
                    "history_answers": datasets.features.Sequence(datasets.Value("string")),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://stanfordnlp.github.io/coqa/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(coqa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = {"train": _TRAIN_DATA_URL, "dev": _DEV_DATA_URL}
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"], "split": "train"}
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"], "split": "validation"}
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["dev"], "split": "test"}
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # TODO(coqa): Yields (key, example) tuples from the dataset

        examples = []

        # 1. Construct conversational question generation triples. [C, [Q], [A]]
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for row in data["data"]:
                questions = [question["input_text"] for question in row["questions"]]
                story = row["story"]
                source = row["source"]
                answers_start = [answer["span_start"] for answer in row["answers"]]
                answers_end = [answer["span_end"] for answer in row["answers"]]
                answers = [answer["input_text"] for answer in row["answers"]]

                example = {
                    "context": story,
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
                if answer_text.lower() == "no" or answer_text.lower() == "yes" or answer_text.lower() == "unknown":
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

        for idx, ex in enumerate(main_examples):
            yield idx, ex
