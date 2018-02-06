from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
import argparse
import functools
import gc
import hashlib

from dataset_utils import load_dataset, clean_text
from features_utils import * # our features
from tfidf import tfidf_features
from pagerank import get_pagerank, get_pagerank_value
from graph_features import get_graph_features
from distance_features import extend_with_features
from multiprocessing import Pool


def create_text_and_graph_features():
    """
    Using the function in feature_utils.py, we create a dataframe with
    text mining features (interrogative, caps, grammatical, leaky features)
    :return: pandas dataframe for train and test set
    """
    ####################################################
    ### Load dataset
    ####################################################
    df_train, df_test = load_dataset()

    ####################################################
    ### Add graph features
    ####################################################
    df_train, df_test = get_graph_features(df_train, df_test)
    df_train_copy, df_test_copy = df_train.copy(), df_test.copy()

    # stopwords
    stops = set(stopwords.words("english"))

    # Add custom features to train/test
    df_train = count_features(df_train)
    df_test = count_features(df_test)

    # questions columns are now list of words for train / test
    df_train['question1'] = df_train['question1'].map(lambda x: clean_text(str(x)).split())
    df_train['question2'] = df_train['question2'].map(lambda x: clean_text(str(x)).split())
    df_test['question1'] = df_test['question1'].map(lambda x: clean_text(str(x)).split())
    df_test['question2'] = df_test['question2'].map(lambda x: clean_text(str(x)).split())

    #List of splitted questions
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist() +
                            df_test['question1'].tolist() + df_test['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    ####################################################
    ### Add word features
    ####################################################
    word_features_train = word_features(df_train, stops, weights)
    df_train = pd.concat((df_train, word_features_train), axis=1)

    word_features_test = word_features(df_test, stops, weights)
    df_test = pd.concat((df_test, word_features_test), axis=1)

    # Find nouns
    train_noun_features = nouns_features(df_train_copy)
    test_noun_features = nouns_features(df_test_copy)
    X_train = pd.concat((df_train, train_noun_features['noun_match']), axis=1)
    X_test = pd.concat((df_test, test_noun_features['noun_match']), axis=1)
    return X_train, X_test

def create_distance_features():
    """
    Common distances
    :return: pandas dataframe for train and test set
    """

    p = Pool(2)
    df_train, df_test = load_dataset()
    df_train, df_test = p.map(extend_with_features, [df_train, df_test])

    X_train = df_train.drop(['id', 'id1', 'id2', "question1", "question2", 'is_duplicate'],
                          axis=1)
    X_test = df_test.drop(['id', 'id1', 'id2', "question1", "question2"],
                          axis=1)
    return X_train, X_test

def create_tfidf_features():
    """
    Using the scikit-learn TFIDF vectorizer, we create a dataframe with some new features.
    We compute the sum, the mean and the length of the TFIDF for both questions.
    :return: pandas dataframe for train and test set
    """
    # Load dataset
    train, test = load_dataset()

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
    _ = tfidf.fit_transform(tfidf_txt)

    trn_features = tfidf_features(train, tfidf)
    tst_features = tfidf_features(test, tfidf)
    # removing unnecessary columns from train and test data
    X_train = trn_features.drop(['id', 'id1', 'id2', "question1", "question2", 'is_duplicate'],
                          axis=1)
    X_test = tst_features.drop(['id', 'id1', 'id2', "question1", "question2"],
                          axis=1)
    return X_train, X_test

def create_pagerank_features():
    """
    We create a dataframe with some features extracted from the PageRank
    algorithm
    :return: pandas dataframe for train and test set
    """
    # Load dataset
    df_train, df_test = load_dataset()

    def generate_qid_graph_table(row):
        """
        Generating a graph of questions and their neighbors.
        Appending nodes to the graph directly
        :param row: dataframe row
        """
        hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
        hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()

        qid_graph.setdefault(hash_key1, []).append(hash_key2)
        qid_graph.setdefault(hash_key2, []).append(hash_key1)

    qid_graph = {}
    _ = df_train.apply(generate_qid_graph_table, axis=1)
    _ = df_test.apply(generate_qid_graph_table, axis=1)
    pagerank_dict = get_pagerank(qid_graph)

    X_train = df_train.apply(lambda x:get_pagerank_value(x, pagerank_dict), axis=1)
    # Empty garbage collector
    del df_train
    gc.collect()
    X_test = df_test.apply(lambda x:get_pagerank_value(x, pagerank_dict), axis=1)
    return X_train, X_test
