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
"""SQUAD: The Stanford Question Answering Dataset."""


import json
import random
import collections
from nltk import word_tokenize, sent_tokenize
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer

import datasets
from datasets.tasks import QuestionAnsweringExtractive

import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": _URL + "train-v1.1.json",
    "dev": _URL + "dev-v1.1.json",
}


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

class Squad(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

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
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        device = torch.device(0)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if self.config.with_history and self.config.history_select_strategy == "DR":
            bert_model = BertModel.from_pretrained("bert-base-uncased")
            bert_model = bert_model.to(device)

        if self.config.with_history and self.config.history_select_strategy == "NSP":
            bert4nsp = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
            bert4nsp = bert4nsp.to(device)

        if self.config.with_history and self.config.with_anaphor:
            predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
                cuda_device=0)

            # https://www.thefreedictionary.com/List-of-pronouns.htm
            pronoun_set = set([word.strip().lower() for word in open("data/pronouns.txt", 'r', encoding='utf-8').readlines()])

        count = 0

        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")

                p_num = len(article["paragraphs"])
                for p_idx, paragraph in enumerate(article["paragraphs"]):
                    context_text = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    context_tokens = word_tokenize(context_text)
                    context_text = " ".join(context_tokens)

                    questions, answers = [], []

                    for qa in paragraph["qas"]:

                        question_text = qa["question"]
                        question_tokens = word_tokenize(question_text.strip())
                        question_text = " ".join(question_tokens)

                        questions.append(question_text)

                        answer_text = qa["answers"][0]['text']
                        answer_tokens = word_tokenize(answer_text.strip())
                        answer_text = " ".join(answer_tokens)

                        answers.append(answer_text)

                    question_num = len(questions)

                    if self.config.with_anaphor:
                        combined_text = context_text + " " + " ".join([
                            question_text + " . " + answer_text + " ."  for question_text, answer_text in zip(questions, answers)])

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
                    for qa_idx, (question_text, answer_text) in enumerate(zip(questions, answers)):

                        # history_num = random.choice([1, 2, 3])
                        history_num = self.config.max_history_turns

                        history_questions, history_answers = [], []

                        candidates = [idx for idx in range(question_num) if not idx == qa_idx and not questions[idx] == questions[qa_idx]]
                        history_qs = [questions[idx] for idx in candidates]
                        followup_q = questions[qa_idx]

                        history_num = min(history_num, len(candidates))

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
                                    sampled_article = random.choice(squad["data"])
                                    sampled_paragraph = random.choice(sampled_article["paragraphs"])

                                    sampled_turn_idx = random.choice(list(range(len(sampled_paragraph["qas"]))))

                                    history_questions += [sampled_paragraph["qas"][sampled_turn_idx]["question"]]
                                    history_answers += [sampled_paragraph["qas"][sampled_turn_idx]["answers"][0]['text']]

                            else:
                                raise NotImplementedError

                            if not self.config.history_select_strategy == "OCR":
                                history_questions = [questions[idx] for idx in history_indices]
                                history_answers = [answers[idx] for idx in history_indices]


                        # CQT
                        if self.config.with_anaphor and history_questions :
                            # Anaphora operation with the latest history
                            # reference = history_questions[-1]
                            reference = " ".join([hq + " . " + ha + " ."  for hq, ha in zip(history_questions, history_answers)])

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

