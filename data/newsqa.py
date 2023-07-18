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
"""This code is used to read and load NewsQA dataset."""

import random
import collections
import re
import numpy
from tqdm import tqdm

from nltk import word_tokenize, sent_tokenize
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer

import csv
import json
import os
from textwrap import dedent

import datasets

import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{trischler2017newsqa,
  title={NewsQA: A Machine Comprehension Dataset},
  author={Trischler, Adam and Wang, Tong and Yuan, Xingdi and Harris, Justin and Sordoni, Alessandro and Bachman, Philip and Suleman, Kaheer},
  booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
  pages={191--200},
  year={2017}
}
"""

# You can copy an official description
_DESCRIPTION = """\
NewsQA is a challenging machine comprehension dataset of over 100,000 human-generated question-answer pairs. \
Crowdworkers supply questions and answers based on a set of over 10,000 news articles from CNN, with answers consisting of spans of text from the corresponding articles.
"""

_HOMEPAGE = "https://www.microsoft.com/en-us/research/project/newsqa-dataset/"

_LICENSE = 'NewsQA Code\
Copyright (c) Microsoft Corporation\
All rights reserved.\
MIT License\
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\
THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\
© 2020 GitHub, Inc.'


class SquadConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self,
                 with_history=False, history_select_strategy="NSP",
                 with_anaphor=False,
                 max_history_turns=20,
                 **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)

        self.with_history = with_history
        self.history_select_strategy = history_select_strategy

        self.with_anaphor = with_anaphor

        self.max_history_turns = max_history_turns


class Newsqa(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")


    BUILDER_CONFIGS = [
        SquadConfig(name="RAW", with_history=False),

        SquadConfig(name="CKT",
                    with_history=True, history_select_strategy="NSP",
                    with_anaphor=True),

        SquadConfig(name="CKT-CTQ",
                    with_history=True, history_select_strategy="NSP",
                    with_anaphor=False),

    ]

    history_select_strategies = ["OCR", "ICR", "TI", "LD", "DR", "NSP"]
    for hs in history_select_strategies:
        BUILDER_CONFIGS += [
            SquadConfig(name=f"CKT+{hs}",
                        with_history=True, history_select_strategy=hs,
                        with_anaphor=True),
        ]

    @property
    def manual_download_instructions(self):
        return dedent(
            """\
            Due to legal restrictions with the CNN data and data extraction. The data has to be downloaded from several sources and compiled as per the instructions by Authors.
            Upon obtaining the resulting data folders, it can be loaded easily using the datasets API.
            Please refer to (https://github.com/Maluuba/newsqa) to download data from Microsoft Reseach site (https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321) and a CNN datasource (https://cs.nyu.edu/~kcho/DMQA/) and run the scripts present here (https://github.com/Maluuba/newsqa).
            This will generate a folder named "split-data" and a file named "combined-newsqa-data-v1.csv".
            Copy the above folder and the file to a directory where you want to store them locally."""
        )

    def _info(self):
        features = datasets.Features(
            {
                "context": datasets.Value("string"),
                "history_questions": datasets.features.Sequence(datasets.Value("string")),
                "history_answers": datasets.features.Sequence(datasets.Value("string")),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        path_to_manual_folder = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if not os.path.exists(path_to_manual_folder):
            raise FileNotFoundError(
                f"{path_to_manual_folder} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('newsqa', data_dir=...)` that includes files from the Manual download instructions: {self.manual_download_instructions}"
            )
        split_files = os.path.join(path_to_manual_folder, "split_data")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(split_files, "train.csv"),
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={"filepath": os.path.join(split_files, "test.csv"), "split": "test"},
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(split_files, "dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        device = torch.device(0)

        if self.config.with_history and self.config.history_select_strategy == "DR":
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            bert_model = BertModel.from_pretrained("bert-base-uncased")
            bert_model = bert_model.to(device)

        if self.config.with_history and self.config.history_select_strategy == "NSP":
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            bert4nsp = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
            bert4nsp = bert4nsp.to(device)

        if self.config.with_history and self.config.with_anaphor:
            predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
                cuda_device=0)

            # https://www.thefreedictionary.com/List-of-pronouns.htm
            pronoun_set = set([word.strip().lower() for word in open("data/pronouns.txt", 'r', encoding='utf-8').readlines()])

        stories = {}

        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                if row:
                    story_id = row[0]

                    story_text = row[1]

                    question_text = row[2].replace("''", '" ').replace("``", '" ')

                    story_tokens = story_text.split()
                    first_answer_span = re.split(',', row[3])[0] if ',' in row[3] else row[3]
                    answer_token_ranges = [int(t) for t in re.split(':', first_answer_span)]
                    answer_text = " ".join(story_tokens[answer_token_ranges[0]:answer_token_ranges[-1]])

                    example = {
                        "context": story_text,
                        "question": question_text,
                        "answer_token_ranges": answer_token_ranges,
                        "answer": answer_text,
                    }

                    if story_id not in stories:
                        stories[story_id] = []

                    stories[story_id].append(example)

        numbers = [len(v) for k, v in stories.items()]
        min_num = min(numbers)
        max_num = max(numbers)
        avg_num = sum(numbers) / len(numbers)

        pct=90
        pct_num = numpy.percentile(sorted(numbers), q=pct)

        print(f" *** {split}: AVG = {avg_num}, MAX = {max_num}, MIN = {min_num}, PCT-{pct} = {pct_num}")

        count = 0
        for _, (story_id, story) in tqdm(enumerate(stories.items()), total=len(stories), desc="Processing Stories ... "):
            story_text = story[0]["context"]

            questions, answers = [], []
            for ex in story:
                questions.append(ex["question"])
                answers.append(ex["answer"])

            question_num = len(story)

            if self.config.with_anaphor:
                combined_text = story_text + " " + " ".join([story[idx]['question'] + " " + story[idx]['answer'] + " ." for idx in range(question_num)])

                result = predictor.predict(document=combined_text)
                tokens = result['document']

                # entity2set = {}
                entity2set = []

                # print('TOKENS: ', result['document'])
                # print('CLUSTERS: ', result['clusters'])
                for cluster in result['clusters']:
                    entity_set = set()
                    for span in cluster:
                        entity = " ".join(tokens[span[0]:span[1] + 1])
                        entity_set.add(entity)

                    for entity in entity_set:
                        # entity2set[entity] = [ent for ent in entity_set if len(entity) > len(ent)]
                        entity2set += [
                            (entity, list(set([ent for ent in entity_set if len(entity) >= len(ent)])))]

                entity2set = collections.OrderedDict(
                    sorted(entity2set, key=lambda x: len(x[0]), reverse=True))

            if self.config.history_select_strategy == "DR":
                def bert_sentence_embedding(text_a):
                    encoding = bert_tokenizer(text_a, padding=True, return_tensors="pt")
                    input_ids = encoding['input_ids'].to(device)
                    token_type_ids = encoding['token_type_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)

                    outputs = bert_model(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask)

                    return outputs.pooler_output

                q_embeddings = bert_sentence_embedding(questions)

                pairwise_similarity = torch.mm(q_embeddings, q_embeddings.T)

            if self.config.history_select_strategy == "TI":
                q_corpus = questions

                vect = TfidfVectorizer(min_df=1, stop_words="english")
                tfidf = vect.fit_transform(q_corpus)
                pairwise_similarity = tfidf * tfidf.T

                pairwise_similarity = pairwise_similarity.toarray()


            distances_mapping = {}
            for qa_idx in range(question_num):

                question_text = questions[qa_idx]
                answer_text = answers[qa_idx]

                story_tokens = story[qa_idx]["context"].split()
                answer_token_ranges = story[qa_idx]['answer_token_ranges']
                pivot = 256
                if answer_token_ranges[0] < pivot:
                    context_text = " ".join(story_tokens[:pivot * 2])
                else:
                    context_text = " ".join(story_tokens[answer_token_ranges[0] - pivot: answer_token_ranges[0] + pivot])
                context_text = context_text.replace("''", '" ').replace("``", '" ')


                history_questions, history_answers = [], []

                candidates = [idx for idx in range(question_num) if not idx == qa_idx and not questions[idx] == questions[qa_idx]]
                history_qs = [questions[idx] for idx in candidates]
                followup_q = questions[qa_idx]

                history_num = min(self.config.max_history_turns, len(candidates))

                if self.config.with_history and candidates:

                    if self.config.history_select_strategy == "ICR":
                        history_indices = random.sample(candidates, history_num)

                    elif self.config.history_select_strategy == "NSP":
                        def next_sentence_prediction(text_a, text_b, topk=3):
                            encoding = bert_tokenizer(text_a, text_b, padding=True, return_tensors="pt")
                            input_ids = encoding['input_ids'].to(device)
                            token_type_ids = encoding['token_type_ids'].to(device)
                            attention_mask = encoding['attention_mask'].to(device)

                            outputs = bert4nsp(input_ids=input_ids,
                                               token_type_ids=token_type_ids,
                                               attention_mask=attention_mask)

                            true_logit = outputs.logits[:, 0]  # True Logit.

                            values, indices = torch.topk(true_logit, k=topk)

                            return indices.tolist()


                        followup_q = [followup_q] * len(history_qs)
                        most_related_idx = next_sentence_prediction(history_qs, followup_q, topk=history_num)

                        history_indices = [candidates[idx] for idx in most_related_idx]

                        history_indices.reverse()

                    elif self.config.history_select_strategy == "DR" or self.config.history_select_strategy == "TI":
                        values = [pairwise_similarity[qa_idx][idx] for idx in candidates]

                        indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:history_num]

                        history_indices = [candidates[idx] for idx in indices]

                        history_indices.reverse()

                    elif self.config.history_select_strategy == "LD":
                        # 1. Ranking
                        distances = []

                        for idx in candidates:
                            name = f"{qa_idx}-{idx}" if qa_idx < idx else f"{idx}-{qa_idx}"
                            if name in distances_mapping:
                                distance = distances_mapping[name]
                            else:
                                distance = edit_distance(questions[idx], questions[qa_idx])
                                distances_mapping[name] = distance

                            distances.append(distance)

                        indices = sorted(range(len(distances)), key=lambda i: distances[i])[:history_num]

                        # 2. Top-k
                        history_indices = [candidates[idx] for idx in indices]

                        history_indices.reverse()

                    elif self.config.history_select_strategy == "OCR":
                        for _ in range(history_num):
                            sampled_story = random.choice(list(stories.items()))[1]

                            sampled_turn_idx = random.choice(list(range(len(sampled_story))))

                            history_questions += [sampled_story[sampled_turn_idx]["question"]]
                            history_answers += [sampled_story[sampled_turn_idx]["answer"]]

                    else:
                        raise NotImplementedError

                    if not self.config.history_select_strategy == "OCR":
                        history_questions = [questions[idx] for idx in history_indices]
                        history_answers = [answers[idx] for idx in history_indices]

                # CQT
                if self.config.with_anaphor and history_questions:
                    # Anaphora operation with the latest history
                    # reference = history_questions[-1]
                    reference = " ".join([hq + " . " + ha + " ." for hq, ha in zip(history_questions, history_answers)])

                    for entity in entity2set:
                        if entity.lower() not in pronoun_set and entity in reference and entity in question_text:
                            entity_group = entity2set[entity]

                            if len(entity_group) > 0:
                                replaced_pronoun = random.choice(entity_group)
                                entity_text = entity
                                question_text = question_text.replace(entity_text, replaced_pronoun)
                                break

                example = {
                    "context": context_text,
                    "history_questions": history_questions,
                    "history_answers": history_answers,
                    "question": question_text,
                    "answer": answer_text,
                }

                yield count, example
                count += 1
