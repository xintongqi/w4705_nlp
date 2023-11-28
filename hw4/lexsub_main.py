#!/usr/bin/env python
import string
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
# import tensorflow

import gensim
import transformers

from typing import List
from collections import defaultdict


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = set()
    syn = wn.synsets(lemma, pos)
    for s in syn:
        for lem in s.lemmas():
            if lem.name() == context.lemma:
                continue
            candidates.add(str(lem.name()).replace('_', ' '))

    return list(candidates)


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    # Part 2
    freq_dic = defaultdict(int)
    syn = wn.synsets(context.lemma, context.pos)
    for s in syn:
        for lem in s.lemmas():
            if lem.name() == context.lemma:
                continue
            freq_dic[str(lem.name()).replace('_', ' ')] += lem.count()

    return max(freq_dic, key=freq_dic.get)


def wn_simple_lesk_predictor(context):
    stop_words = set(stopwords.words('english'))

    target_word = context.lemma
    context_tokens = [token for token in context.left_context + context.right_context if
                      token not in stop_words]

    overlap_list = defaultdict(int)
    for lemma in wn.lemmas(context.lemma, context.pos):
        syn = lemma.synset()
        definition = syn.definition()
        examples = syn.examples()
        hypernyms = syn.hypernyms()

        for ex in examples:
            definition += " " + ex

        for hypernym in hypernyms:
            definition += " " + hypernym.definition()
            for hypernym_ex in hypernym.examples():
                definition += " " + hypernym_ex

        tokens_def = tokenize(definition)
        definition_filtered = [word for word in tokens_def if word not in stop_words]

        overlap = len(set(context_tokens) & set(definition_filtered))
        overlap_list[lemma] = overlap

    max_overlap = max(overlap_list.values())
    best_candidates = [lemma for lemma, overlap in overlap_list.items() if overlap == max_overlap]

    if best_candidates:
        counts = {l.name(): l.count() for lemma in best_candidates for l in lemma.synset().lemmas()}
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        for candidate in sorted_counts.keys():
            if candidate != target_word:
                return candidate.replace('_', ' ')

    return target_word


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        similarity_scores = {}
        candidates = get_candidates(context.lemma, context.pos)

        for candidate in candidates:
            try:
                similarity_scores[candidate] = self.model.similarity(context.lemma, candidate)
            except KeyError:
                continue

        # sorted_similarity_scores = dict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True))
        # most_similar_candidate = list(sorted_similarity_scores.keys())[0] if sorted_similarity_scores else None
        most_similar_candidate = max(similarity_scores, key=similarity_scores.get)
        return most_similar_candidate

    def predict_nearest_improved(self, context: Context) -> str:
        similarity_scores = {}
        candidates = get_candidates(context.lemma, context.pos)

        for candidate in candidates:
            try:
                similarity_scores[candidate] = self.model.similarity(context.lemma, candidate) / (
                        self.model.wv[context.lemma].norm() * self.model.wv[candidate].norm()
                )
                # context_vector = self.model[context.lemma] / self.model[context.lemma].sum()
                # candidate_vector = self.model[candidate] / self.model[candidate].sum()
                # similarity_scores[candidate] = context_vector.dot(candidate_vector)
            except KeyError:
                continue

        most_similar_candidate = max(similarity_scores, key=similarity_scores.get)
        return most_similar_candidate


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        left_ctxt = ' '.join([i if i.isalpha() else i for i in context.left_context])
        right_ctxt = ' '.join([j if j.isalpha() else j for j in context.right_context])
        ctxt = f"{left_ctxt} [MASK] {right_ctxt}"

        input_toks = self.tokenizer.encode(ctxt)
        sent_tokenized = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_idx = sent_tokenized.index('[MASK]')

        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]

        best_probs = np.argsort(predictions[0][mask_idx])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_probs)

        return next((word.replace('_', ' ') for word in best_words if word.replace('_', ' ') in candidates), '')


if __name__ == "__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # # Smurf_predictor
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = smurf_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # # Test part 1
    # print(get_candidates('slow', 'a'))

    # # Test part 2 [0.098/0.136]
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = wn_frequency_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # # Test part 3 [0.102/0.136]
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = wn_simple_lesk_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # Test part 4
    for context in read_lexsub_xml(sys.argv[1]):
        model = Word2VecSubst('GoogleNews-vectors-negative300.bin.gz')
        prediction = model.predict_nearest(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # # Test part 5 [0.115/0.170]
    # for context in read_lexsub_xml(sys.argv[1]):
    #     model = BertPredictor()
    #     prediction = model.predict(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # # Test part 6
    # for context in read_lexsub_xml(sys.argv[1]):
    #     model = Word2VecSubst('GoogleNews-vectors-negative300.bin.gz')
    #     prediction = model.predict_nearest_improved(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
