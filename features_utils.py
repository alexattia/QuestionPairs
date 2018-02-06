import argparse
import functools

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.optimize import minimize

import multiprocessing
import difflib
import datetime
import operator

import gc
import nltk
import pandas as pd



####################################################
# Counts on unprocessed dataset
####################################################

# Counting the number of words in raw sentence.
def add_word_count(x, df, word):
    """
    Counting the number of times a word appears
    :param x: dataframe row
    :param df: dataframe
    :param word: word to count
    """
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word] #TODO + ou * ou min ou max ???
    return x


def count_features(df_features):
    """
    Add new features to the dataframe :
    - Capital letters count
    - Interrogative words
    :param df_features: dataframe
    :return: updated dataframe
    """
    df_features['caps_count_q1'] = df_features['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    df_features['caps_count_q2'] = df_features['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
    df_features['diff_caps'] = df_features['caps_count_q1'] - df_features['caps_count_q2']

    #TODO CLEANING BEFORE OR AFTER ?
    df_features = add_word_count(df_features, df_features,'how')
    df_features = add_word_count(df_features, df_features,'what')
    df_features = add_word_count(df_features, df_features,'which')
    df_features = add_word_count(df_features, df_features,'who')
    df_features = add_word_count(df_features, df_features,'where')
    df_features = add_word_count(df_features, df_features,'when')
    df_features = add_word_count(df_features, df_features,'why')
    return df_features.fillna(0.0)


####################################################
### Word Features
####################################################

def common_words(row):
    """
    Compute the number of common words using the intersection of
    unique wordsbetween the paired questions
    :param row: dataframe row
    :return: number of common words as int
    """
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row, stops=None):
    """
    Compute the total of unique words using the union of
    unique words between the paired questions
    :param row: dataframe row
    :param stops: stop words (optional feature)
    :return: number of unique words as int
    """
    if not stops:
        return len(set(row['question1']).union(row['question2']))
    else:
        return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def total_unq_words_stop(row, stops):
    return total_unique_words(row, stops=stops)

def wc_diff(row, unique=False, stops=None):
    """
    Computing the difference of sentence length between the
    paired questions
    :param row: dataframe row
    :param unique: boolean to consider only unique words
    :param stops: stop words (optional feature)
    :return: number of the difference of word count as int
    """
    if unique:
        return abs(len(set(row['question1'])) - len(set(row['question2'])))
    elif stops:
        return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))
    else:
        return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row, unique=False, stops=None):
    """
    Computing the ratio of the number of words between the
    paired questions
    :param row: dataframe row
    :param unique: boolean to consider only unique words
    :param stops: stop words (optional feature)
    :return: ratio of the number of words as float
    """
    if unique:
        l1 = len(set(row['question1'])) * 1.0
        l2 = len(set(row['question2']))
    elif stops:
        l1 = len([x for x in set(row['question1']) if x not in stops])*1.0
        l2 = len([x for x in set(row['question2']) if x not in stops])
    else:
        l1 = len(row['question1'])*1.0
        l2 = len(row['question2'])
    if l2 == 0.:
        return np.nan
    if l1 / l2 > 1.: #TODO VERIFIER NORMALISATION RATIO
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return wc_diff(row, unique=True)

def wc_ratio_unique(row):
    return wc_ratio(row, unique=True)

def wc_diff_unique_stop(row, stops=None):
    return wc_diff(row, stops=stops)

def wc_ratio_unique_stop(row, stops=None):
    return wc_ratio(row, stops=stops)

def same_start_word(row):
    """
    Check if the paired questions have the same first word
    :param row: dataframe row
    :return: boolean as int
    """
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row, stops=None):
    """
    Compute the difference of length (characters sensitive) of the paired
    questions
    :param row: dataframe row
    :param stops: stop words (optional feature)
    :return: length difference as int
    """
    if stops:
        return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))
    else:
        return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    """
    Compute the ratio of length (characters sensitive) of the paired
    questions
    :param row: dataframe row
    :return: length ratio as float
    """
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0.:
        return np.nan
    if l1 / l2 > 1.: #TODO VERIFEIR
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return char_diff(row, stops=stops)


def diff_ratios(st1, st2):
    """
    Finding the longest contiguous matching subsequence
    (Ratcliff and Metzener)
    :param st1, st2: strings
    :return: float ratio
    """
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()


##########################
### Td-Idf Features
##########################

def get_weight(count, eps=10000., min_count=2.):
    count = float(count)
    if count < min_count:
        return 0
    else:
        return 1. / (count + eps)

def tfidf_word_match_share(row, weights=None, stops=None):
    """
    Compute the ratio of the tf-idf shared words
    :param stops: stop words (optional features)
    :return: ratio of shared weights over the sum of total weights
    """
    q1words = {}
    q2words = {}
    if stops:
        for word in row['question1']:
            if word not in stops:
                q1words[word] = 1
        for word in row['question2']:
            if word not in stops:
                q2words[word] = 1
    else:
        for word in row['question1']:
            q1words[word] = 1
        for word in row['question2']:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share_stops(row, stops=None, weights=None):
    return tfidf_word_match_share(row, weights=weights, stops=stops)


def word_features(data, stops, weights):
    """
    Add first features to the dataframe :
    - Word counts and ratio
    - Characters sensitive counts
    :param data: dataframe
    :param stops: stop words
    :param weights: weights
    :return: updated dataframe
    """
    X = pd.DataFrame()

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    f = functools.partial(char_diff_unique_stop, stops=stops)
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

    X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16

    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17

    return X



####################################################
# Nouns features using POS Tagging.
####################################################
def nouns_features(df_features):
	"""
	Add new features to the dataframe :
	- Number of nouns
	- Number of nouns in both questions
	:param df_features: dataframe
	:return: updated dataframe
	"""

	X = pd.DataFrame(index = df_features.index)
	X['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
	X['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
	X['noun_match'] = X.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)
	return X.fillna(0.0)
