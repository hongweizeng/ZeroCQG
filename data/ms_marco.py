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
"""MS MARCO dataset."""
import random
import re
import json
import collections
from collections import defaultdict, Counter

import math
import numpy
from tqdm import tqdm

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

import datasets
import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction


_CITATION = """
@article{DBLP:journals/corr/NguyenRSGTMD16,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  journal   = {CoRR},
  volume    = {abs/1611.09268},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.09268},
  archivePrefix = {arXiv},
  eprint    = {1611.09268},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}
"""

_DESCRIPTION = """
Starting with a paper released at NIPS 2016, MS MARCO is a collection of datasets focused on deep learning in search.
The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer.
Since then we released a 1,000,000 question dataset, a natural langauge generation dataset, a passage ranking dataset,
keyphrase extraction dataset, crawling dataset, and a conversational search.
There have been 277 submissions. 20 KeyPhrase Extraction submissions, 87 passage ranking submissions, 0 document ranking
submissions, 73 QnA V2 submissions, 82 NLGEN submisions, and 15 QnA V1 submissions
This data comes in three tasks/forms: Original QnA dataset(v1.1), Question Answering(v2.1), Natural Language Generation(v2.1).
The original question answering datset featured 100,000 examples and was released in 2016. Leaderboard is now closed but data is availible below.
The current competitive tasks are Question Answering and Natural Language Generation. Question Answering features over 1,000,000 queries and
is much like the original QnA dataset but bigger and with higher quality. The Natural Language Generation dataset features 180,000 examples and
builds upon the QnA dataset to deliver answers that could be spoken by a smart speaker.
"""
_V2_URLS = {
    "train": "https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz",
    "dev": "https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz",
    "test": "https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz",
}

_V1_URLS = {
    "train": "https://msmarco.blob.core.windows.net/msmsarcov1/train_v1.1.json.gz",
    "dev": "https://msmarco.blob.core.windows.net/msmsarcov1/dev_v1.1.json.gz",
    "test": "https://msmarco.blob.core.windows.net/msmsarcov1/test_hidden_v1.1.json",
}


class MsMarcoConfig(datasets.BuilderConfig):
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
        super(MsMarcoConfig, self).__init__(version=datasets.Version("1.1.0"), **kwargs)

        self.with_history = with_history
        self.history_select_strategy = history_select_strategy

        self.with_anaphor = with_anaphor

        self.max_history_turns = max_history_turns



class MsMarco(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        MsMarcoConfig(name="RAW", with_history=False),

        MsMarcoConfig(name="CKT",
                    with_history=True, history_select_strategy="NSP",
                    with_anaphor=True),

        MsMarcoConfig(name="CKT-CTQ",
                    with_history=True, history_select_strategy="NSP",
                    with_anaphor=False),

    ]

    history_select_strategies = ["OCR", "TI", "LD", "DR", "NSP"]
    for hs in history_select_strategies:
        BUILDER_CONFIGS += [
            MsMarcoConfig(name=f"CKT+{hs}",
                        with_history=True, history_select_strategy=hs,
                        with_anaphor=True),
        ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + "\n" + str(self.config.version),
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "history_questions": datasets.features.Sequence(datasets.Value("string")),
                    "history_answers": datasets.features.Sequence(datasets.Value("string")),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            homepage="https://microsoft.github.io/msmarco/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.version == datasets.Version("2.1.0"):
            dl_path = dl_manager.download_and_extract(_V2_URLS)
        else:
            dl_path = dl_manager.download_and_extract(_V1_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": dl_path["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": dl_path["dev"]},
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={"filepath": dl_path["test"]},
            # ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        if self.config.with_anaphor:

            predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
                cuda_device=0)

            # https://www.thefreedictionary.com/List-of-pronouns.htm
            pronoun_set = set([word.strip().lower() for word in open("data/pronouns.txt", 'r', encoding='utf-8').readlines()])

            proposition_set = set([word.strip().lower().split()[0] for word in open("data/pronouns.txt", 'r', encoding='utf-8').readlines()])


        db = Database()
        index = InvertedIndex(db, name="context")

        examples = []
        count = 0

        with open(filepath, encoding="utf-8") as f:
            if self.config.version == datasets.Version("2.1.0"):
                data = json.load(f)
                questions = data["query"]
                answers = data.get("answers", {})
                passages = data["passages"]
                query_ids = data["query_id"]
                query_types = data["query_type"]
                wellFormedAnswers = data.get("wellFormedAnswers", {})
                for key in questions:

                    is_selected = [passage.get("is_selected", -1) for passage in passages[key]]
                    passage_text = [passage["passage_text"] for passage in passages[key]]
                    urls = [passage["url"] for passage in passages[key]]
                    question = questions[key]
                    answer = answers.get(key, [])
                    query_id = query_ids[key]
                    query_type = query_types[key]
                    wellFormedAnswer = wellFormedAnswers.get(key, [])
                    if wellFormedAnswer == "[]":
                        wellFormedAnswer = []
                    # yield query_id, {
                    #     "answers": answer,
                    #     "passages": {"is_selected": is_selected, "passage_text": passage_text, "url": urls},
                    #     "query": question,
                    #     "query_id": query_id,
                    #     "query_type": query_type,
                    #     "wellFormedAnswers": wellFormedAnswer,
                    # }

                    if any(is_selected):
                        selected_idx = [idx for idx, tag in enumerate(is_selected) if not tag == 0][0]
                        context_text = passage_text[selected_idx]
                        question_text = question
                        answer_text = answer[0] if len(answer) else ""

                        if answer_text and not answer_text.lower() == "yes" and not answer_text.lower() == 'no':

                            example = {
                                'id': count,
                                'context': context_text,
                                'question': question_text,
                                'answer': answer_text
                            }
                            count += 1
                            examples.append(example)
                            # index.index_example(example)

            if self.config.version == datasets.Version("1.1.0"):
                for row in f:
                    data = json.loads(row)
                    question = data["query"]
                    answer = data.get("answers", [])
                    passages = data["passages"]
                    query_id = data["query_id"]
                    query_type = data["query_type"]
                    wellFormedAnswer = data.get("wellFormedAnswers", [])

                    is_selected = [passage.get("is_selected", -1) for passage in passages]
                    passage_text = [passage["passage_text"] for passage in passages]
                    urls = [passage["url"] for passage in passages]
                    if wellFormedAnswer == "[]":
                        wellFormedAnswer = []
                    # yield query_id, {
                    #     "answers": answer,
                    #     "passages": {"is_selected": is_selected, "passage_text": passage_text, "url": urls},
                    #     "query": question,
                    #     "query_id": query_id,
                    #     "query_type": query_type,
                    #     "wellFormedAnswers": wellFormedAnswer,
                    # }

                    if any(is_selected):
                        selected_idx = [idx for idx, tag in enumerate(is_selected) if tag == 1][0]
                        context_text = passage_text[selected_idx]
                        question_text = question
                        answer_text = answer[0] if len(answer) else ""

                        if answer_text and not answer_text.lower() == "yes" and not answer_text.lower() == 'no':

                            context_tokens = word_tokenize(context_text.strip())
                            context_text = " ".join(context_tokens)

                            question_tokens = word_tokenize(question_text.strip())
                            question_text = " ".join(question_tokens)

                            answer_tokens = word_tokenize(answer_text.strip())
                            answer_text = " ".join(answer_tokens)

                            example = {
                                'id': count,
                                'context': context_text,
                                'question': question_text,
                                'answer': answer_text
                            }
                            count += 1
                            examples.append(example)
                            index.index_example(example)

                            # if count % 100 == 0:
                            #     break

        print("Init IDF ... ")
        index.init_idf()

        ex_num = len(examples)

        def calculate_tfidf_similarity(column_name):
            corpus = [ex[column_name] for ex in examples]
            vect = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf = vect.fit_transform(corpus)
            pair_similarity = tfidf * tfidf.T
            pair_similarity = pair_similarity.toarray()
            return pair_similarity

        def calculate_dr_similarity(colum_name, tokenizer, model):
            batch_size = 32
            num_batch = math.ceil(ex_num / batch_size)

            dense_vectors = []

            for batch_idx in tqdm(range(num_batch), desc="[Building DR] ->"):
                batch_q =  [ex[colum_name] for ex in examples[batch_size * batch_idx: batch_size * (batch_idx + 1)]]

                encoding = tokenizer(batch_q, padding=True, return_tensors="pt")
                # outputs = bert_model(**encoding)

                input_ids = encoding['input_ids'].to(device)
                token_type_ids = encoding['token_type_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

                batch_dense = outputs.pooler_output

                batch_dense = batch_dense.detach().cpu().numpy()

                dense_vectors.append(batch_dense)

            dense_vectors = numpy.concatenate(dense_vectors, axis=0)

            pair_similarity = numpy.matmul(dense_vectors, dense_vectors.T)
            return pair_similarity

        if self.config.with_history and self.config.max_history_turns:
            use_tfidf = True
            # use_tfidf = False
            if use_tfidf:
                index.context_pairwise_similarity = calculate_tfidf_similarity(column_name="context")
                print("CONTEXT TF-IDF finished!")

            else:
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

                device = torch.device(0)
                bert_model = BertModel.from_pretrained("bert-base-uncased")
                bert_model = bert_model.to(device)

                index.context_pairwise_similarity = calculate_dr_similarity(colum_name="context",
                                                                            tokenizer=bert_tokenizer,
                                                                            model=bert_model)
                print("CONTEXT Dense Retrieval finished!")

            if self.config.history_select_strategy == "TI":
                index.pairwise_similarity = calculate_tfidf_similarity(column_name="question")
                print("QUESTION TF-IDF finished!")

            if self.config.history_select_strategy == "DR":
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

                device = torch.device(0)
                bert_model = BertModel.from_pretrained("bert-base-uncased")
                bert_model = bert_model.to(device)

                index.pairwise_similarity = calculate_dr_similarity(colum_name="question",
                                                                    tokenizer=bert_tokenizer,
                                                                    model=bert_model)

                print("QUESTION Dense Retrieval finished!")

            if self.config.history_select_strategy == "NSP":
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                device = torch.device(0)
                bert4nsp = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
                bert4nsp = bert4nsp.to(device)

                index.bert_tokenizer = bert_tokenizer
                index.bert4nsp = bert4nsp


        count = 0

        for qa_idx, example in tqdm(enumerate(examples), total=ex_num, desc="Processing ... "):
            context_text = example['context']
            question_text = example['question']
            answer_text = example['answer']

            history_questions, history_answers = [], []
            if self.config.with_history and self.config.max_history_turns:

                if self.config.history_select_strategy == "OCR":
                    candidates = [idx for idx in range(ex_num) if not idx == qa_idx]
                    history_indices = random.sample(candidates, self.config.max_history_turns)
                else:
                    history_indices = index.lookup_query(example,
                        ranking_algorithm=self.config.history_select_strategy,
                        max_history_turns=self.config.max_history_turns)

                for h_id in history_indices:
                    h_ex = db.get(h_id)
                    history_questions.append(h_ex['question'])
                    history_answers.append(h_ex['answer'])

                # CQT
                if self.config.with_anaphor and history_questions:
                    combined_text = context_text + " " + " ".join([
                        q_text + " . " + a_text + " ." for q_text, a_text in zip(history_questions, history_answers)])

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

            yield count, {
                "context": context_text,
                "history_questions": history_questions,
                "history_answers": history_answers,
                "question": question_text,
                "answer": answer_text,
            }
            count += 1



# https://medium.com/@fro_g/writing-a-simple-inverted-index-in-python-3c8bcb52169a
class Appearance:
    """
    Represents the appearance of a term in a given document, along with the
    frequency of appearances in the same one.
    """

    def __init__(self, id, frequency):
        self.id = id
        self.frequency = frequency

    def __repr__(self):
        """
        String representation of the Appearance object
        """
        return str(self.__dict__)


class Database:
    """
    In memory database representing the already indexed documents.
    """

    def __init__(self):
        self.db = dict()

    def __repr__(self):
        """
        String representation of the Database object
        """
        return str(self.__dict__)

    def __len__(self):
        return len(self.db)

    def get(self, id):
        return self.db.get(id, None)

    def add(self, example):
        """
        Adds a document to the DB.
        """
        return self.db.update({example['id']: example})

    def remove(self, example):
        """
        Removes document from DB.
        """
        return self.db.pop(example['id'], None)


class InvertedIndex:
    """
    Inverted Index class.
    """

    def __init__(self, db, name="context", max_coarse_num=100, max_fine_grained_num=20,
                 context_pairwise_similarity=None, pairwise_similarity=None,
                 bert_tokenizer=None, bert4nsp=None):
        self.index = dict()
        self.db = db
        self.name = name

        self.stopwords = set(stopwords.words('english'))
        self.stemmer = snowball.SnowballStemmer('english')

        self.distances = dict()

        self.idf = {}

        self.max_coarse_num = max_coarse_num
        self.max_fine_grained_num = max_fine_grained_num

        self.context_pairwise_similarity = context_pairwise_similarity
        self.pairwise_similarity = pairwise_similarity

        self.bert_tokenizer = bert_tokenizer
        self.bert4nsp = bert4nsp


    def init_idf(self):
        doc_number = len(self.db)

        for key, val in self.index.items():
            self.idf[key] = numpy.log(doc_number / (len(val) + 1))

    def __repr__(self):
        """
        String representation of the Database object
        """
        return str(self.index)

    def process_text(self, text):
        # Remove punctuation from the text.
        clean_text = re.sub(r'[^\w\s]', '', text)

        words = word_tokenize(clean_text)

        # Remove stopwords from the text.
        non_stopwords = set(words) - set(stopwords.words('english'))

        terms = [self.stemmer.stem(x) for x in non_stopwords]

        return terms

    def index_example(self, example):
        """
        Process a given document, save it to the DB and update the index.
        """

        text = self.ex2text(example)
        terms = self.process_text(text)
        terms = set(terms)

        appearances_dict = dict()
        # Dictionary with each term and the frequency it appears in the text.
        for term in terms:
            term_frequency = appearances_dict[term].frequency if term in appearances_dict else 0
            appearances_dict[term] = Appearance(example['id'], term_frequency + 1)

        # Update the inverted index
        update_dict = {key: [appearance] if key not in self.index else self.index[key] + [appearance]
                       for (key, appearance) in appearances_dict.items()}

        self.index.update(update_dict)
        # Add the document into the database
        self.db.add(example)

        return example

    def ex2text(self, example):
        if self.name == "context":
            text = example['context']
        elif self.name == "question":
            text = example['question']
        elif self.name == "context+question":
            text = example['context'] + " " + example['question']
        else:
            text = example['context']

        return text

    def lookup_query(self, query, ranking_algorithm="TI", max_history_turns=3):
        """
        Returns the dictionary of terms with their correspondent Appearances.
        This is a very naive search since it will just split the terms and show
        the documents where they appear.
        """

        query_id = query['id']
        query_question = query['question']

        max_inverted_index_num = 1000
        max_context_num = 100

        # 1. Inverted Index.
        query_text = self.ex2text(query)
        query_terms = self.process_text(query_text)
        # query_terms = set(query_terms)

        scores = defaultdict(int)
        for term in query_terms:
            if term in self.index:
                for appearance in self.index[term]:
                    if not appearance.id == query_id:
                        # scores[appearance.id] += appearance.frequency
                        scores[appearance.id] += 1

        coarse_history_indices = sorted(scores, key=lambda x: scores[x], reverse=True)
        candidates = coarse_history_indices[:max_inverted_index_num]

        candidates = [idx for idx in candidates if not idx == query_id and
                      not self.db.get(idx)["question"] == self.db.get(query_id)["question"] and
                      not self.db.get(idx)["answer"] == self.db.get(query_id)["answer"]]

        # 2. Context Similarity Ranking.
        ctx_similarities = [self.context_pairwise_similarity[query_id][idx] for idx in candidates]
        indices = sorted(range(len(ctx_similarities)), key=lambda i: ctx_similarities[i], reverse=True)[:max_context_num]
        candidates = [candidates[idx] for idx in indices]
        candidates.reverse()

        # 2. Question Similarity Ranking.
        if ranking_algorithm == "TI" or ranking_algorithm == "DR":
            similarities = [self.pairwise_similarity[query_id][idx] for idx in candidates]

            indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:max_history_turns]

            history_indices = [candidates[idx] for idx in indices]

            history_indices.reverse()

        elif ranking_algorithm == "LD":
            # 1. Ranking
            distances = []

            for idx in candidates:
                name = f"{query_id}-{idx}" if query_id < idx else f"{idx}-{query_id}"
                if name in self.distances:
                    distance = self.distances[name]
                else:
                    distance = edit_distance(self.db.get(idx)["question"], self.db.get(query_id)["question"])
                    self.distances[name] = distance

                distances.append(distance)

            indices = sorted(range(len(distances)), key=lambda i: distances[i])[:max_history_turns]

            # 2. Top-k
            history_indices = [candidates[idx] for idx in indices]

            history_indices.reverse()

        elif ranking_algorithm == "NSP":

            def next_sentence_prediction(text_a, text_b, topk):
                encoding = self.bert_tokenizer(text_a, text_b, padding=True, return_tensors="pt")

                device = next(self.bert4nsp.parameters()).device

                input_ids = encoding['input_ids'].to(device)
                token_type_ids = encoding['token_type_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = self.bert4nsp(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)

                true_logit = outputs.logits[:, 0]  # True Logit.

                values, indices = torch.topk(true_logit, k=topk)

                return indices.tolist()

            history_qs = [self.db.get(idx)["question"] for idx in candidates]
            followup_q = [query_question] * len(candidates)

            most_related_idx = next_sentence_prediction(history_qs, followup_q, topk=max_history_turns)

            history_indices = [candidates[idx] for idx in most_related_idx]

            history_indices.reverse()

        else:
            raise NotImplementedError

        return history_indices
