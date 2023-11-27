#!/usr/bin/env python
import string
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

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


def wn_simple_lesk_predictor(context: Context) -> str:
    # Part 3
    # remove stop words
    stop_words = stopwords.words('english')
    con = context.left_context + context.right_context
    filtered_con = set([x for x in con if x not in stop_words])

    # count overlaps
    overlap_dict = defaultdict()
    lemmas = wn.lemmas(context.lemma, context.pos)
    for lem in lemmas:
        s = lem.synset()
        # get definition of the synset
        definition = s.definition()
        # add all examples
        for e in s.examples():
            definition += (' ' + e)
        # add all hypernyms
        for h in s.hyponyms():
            definition += (' ' + h.definition())
        # process definition into dictionary
        tokenized_def = tokenize(definition)
        filtered_def = set([x for x in tokenized_def if x not in stop_words])
        overlap_dict[lem] = len(filtered_con & filtered_def)

    # return values
    max_overlap_value = max(overlap_dict.values(), default=0)
    max_count = defaultdict(int)
    max_synset = None

    if max_overlap_value > 0:
        max_overlap = max(overlap_dict, key=overlap_dict.get)
        max_synset = max_overlap.synset()
    else:
        lemma_count = defaultdict(int)
        for syn in wn.synsets(context.lemma, context.pos):
            for lemma in syn.lemmas():
                if lemma.name() == context.lemma:
                    continue
                lemma_count[lemma] += lemma.count()
        if lemma_count:
            max_lemma = max(lemma_count, key=lemma_count.get)
            max_synset = max_lemma.synset()

    if max_synset:
        max_count = defaultdict(int)
        for lemma in max_synset.lemmas():
            if lemma.name() == context.lemma:
                continue
            max_count[lemma.name()] += lemma.count()

    return max(max_count, key=max_count.get, default='defaultsetting')


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        return None  # replace for part 4


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        return None  # replace for part 5


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

    # # Test part 2
    # for context in read_lexsub_xml(sys.argv[1]):
    #     # print(context)  # useful for debugging
    #     prediction = wn_frequency_predictor(context)
    #     print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # Test part 3
    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        prediction = wn_simple_lesk_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
