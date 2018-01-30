from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
import argparse
import functools
# Our features
from features_utils import *

def create_preprocessed_features():
    """
    Using the function in feature_utils.py, we create a dataframe with
    text mining features (interrogative, caps, grammatical, leaky features)
    :return: pandas dataframe for train and test set
    """
    # Load dataset
    df_train, df_test = load_dataset()

    # Create a dictionnary of paired questions for every question
    ques = pd.concat([df_train[['question1', 'question2']], \
        df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
            q_dict[ques.question1[i]].add(ques.question2[i])
            q_dict[ques.question2[i]].add(ques.question1[i])
            
    # Frequency features 
    df_train['q1_q2_intersect'] = df_train.apply(lambda x: q1_q2_intersect(x, q_dict), axis=1, raw=True)
    df_train['q1_freq'] = df_train.apply(lambda x: q1_freq(x, q_dict), axis=1, raw=True)
    df_train['q2_freq'] = df_train.apply(lambda x: q2_freq(x, q_dict), axis=1, raw=True)

    df_test['q1_q2_intersect'] = df_test.apply(lambda x: q1_q2_intersect(x, q_dict), axis=1, raw=True)
    df_test['q1_freq'] = df_test.apply(lambda x: q1_freq(x, q_dict), axis=1, raw=True)
    df_test['q2_freq'] = df_test.apply(lambda x: q2_freq(x, q_dict), axis=1, raw=True)

    test_leaky = df_test.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]
    train_leaky = df_train.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]

    # Add other custom features to train
    stops = set(stopwords.words("english"))

    ## TODO REPLACE SPLIT BY PREPROCESSED TEXT
    df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
    df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    X_train = build_features(df_train, stops, weights)
    X_train = pd.concat((X_train, train_leaky), axis=1)

    # Add other custom features to test
    df_test = pd.read_csv('test.csv', header=None)
    df_test.columns = ['id', 'id1', 'id2', 'question1', 'question2']
    df_test = df_test.fillna(' ')

    df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
    df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())
    X_test = build_features(df_test, stops, weights)
    X_test = pd.concat((X_test, test_leaky), axis=1)
    return X_train, X_test

def create_tfidf_features():
    """
    Using the scikit-learn TFIDF vectorizer, we create a dataframe with some new features.
    We compute the sum, the mean and the length of the TFIDF for both questions.
    :return: pandas dataframe for train and test set
    """
    # Load dataset
    train, test = load_dataset()

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))

    tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
    _ = tfidf.fit_transform(tfidf_txt)
    _ = cvect.fit_transform(tfidf_txt)

    trn_features = get_features(train, tfidf)
    tst_features = get_features(test, tfidf)

    trn_features = get_noun(trn_features)
    tst_features = get_noun(tst_features)

    # removing unnecessary columns from train and test data
    X_train = trn_features.iloc[:,8:]
    X_test = tst_features.iloc[:,7:]
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
    pagerank_dict = pagerank(qid_graph)

    X_train = df_train.apply(lambda x:get_pagerank_value(x, pagerank_dict), axis=1)
    # Empty garbage collector
    del df_train
    gc.collect()
    X_test = df_test.apply(lambda x:get_pagerank_value(x, pagerank_dict), axis=1)
    return X_train, X_test
