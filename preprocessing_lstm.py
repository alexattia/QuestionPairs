import os
import re
import csv
import codecs
# data
import numpy as np
import pandas as pd
# text preprocessing
from string import punctuation
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 40
VALIDATION_SPLIT = 0.1
GLOVE_DIR = './'
EMBEDDING_DIM = 300


def tokenization(texts_1, texts_2, test_texts_1, test_texts_2):
    # tokenize
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
    word_index = tokenizer.word_index
    return sequences_1, sequences_2, test_sequences_1, test_sequences_2, word_index

def preprocessing():
    # preprocessing
    print("-- Preprocessing Start")
    df_train = pd.read_csv('train.csv', header=None)
    df_train = df_train.fillna(' ')
    df_train.columns = ['id', 'id1', 'id2', 'question1', 'question2', 'is_duplicate']
    texts_1 = df_train["question1"].values.tolist()
    texts_2 = df_train["question2"].values.tolist()
    labels = df_train['is_duplicate'].apply(int)
    labels = np.array(labels)
    print("-- Preprocessing - train completed")
    df_test = pd.read_csv('test.csv', header=None)
    df_test = df_test.fillna(' ')
    df_test.columns = ['id', 'id1', 'id2', 'question1', 'question2']
    test_texts_1 = df_test["question1"].values.tolist()
    test_texts_2 = df_test["question2"].values.tolist()
    test_ids = df_test["id"]
    print("-- Preprocessing - test completed")
    print("-- Preprocessing - tokenization")
    sequences_1, sequences_2, test_sequences_1, test_sequences_2, word_index = tokenization(texts_1, texts_2, test_texts_1, test_texts_2)
    print("-- Preprocessing - Sentence Truncate")
    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH) #padding : add values at the end to compare phrases from different lengths
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)
    return data_1, data_2, test_data_1, test_data_2, test_ids, labels, word_index

def load_leaky():
    # new features
    print("-- Preprocessing - load leaky features")
    trn = pd.read_csv('X_train.csv', index_col=0)
    trn = trn.drop(["is_duplicate","question1", "question2"], axis=1) #,"question1_nouns", "question2_nouns"
    tst = pd.read_csv('X_test.csv', index_col=0)
    tst = tst.drop(["question1", "question2"], axis=1) #,"question1_nouns", "question2_nouns"
    trn = trn.replace([np.inf, -np.inf], np.nan)
    tst = tst.replace([np.inf, -np.inf], np.nan)
    trn = trn.fillna(value=0)
    tst = tst.fillna(value=0)
    trn.shape, tst.shape
    print(trn.isnull().values.any())
    print(tst.isnull().values.any())
    print("-- Preprocessing - rescale leaky features")
    leaks = trn[trn.columns.values]
    test_leaks = tst[tst.columns.values]
    ss = StandardScaler()
    ss.fit(np.vstack((leaks, test_leaks)))
    leaks = ss.transform(leaks)
    test_leaks = ss.transform(test_leaks)
    return leaks, test_leaks

def input_nn_data(data_1, data_2, test_data_1, test_data_2, test_ids, labels, leaks, test_leaks):
    np.random.seed(1234)
    perm = np.random.permutation(len(data_1))
    idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
    idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]
    data = {}
    data["data_1_train"] = np.vstack((data_1[idx_train], data_2[idx_train]))
    data["data_2_train"] = np.vstack((data_2[idx_train], data_1[idx_train]))
    data["leaks_train"] = np.vstack((leaks[idx_train], leaks[idx_train]))
    data["labels_train"] = np.concatenate((labels[idx_train], labels[idx_train]))

    data["data_1_val"] = np.vstack((data_1[idx_val], data_2[idx_val]))
    data["data_2_val"] = np.vstack((data_2[idx_val], data_1[idx_val]))
    data["leaks_val"] = np.vstack((leaks[idx_val], leaks[idx_val]))
    data["labels_val"] = np.concatenate((labels[idx_val], labels[idx_val]))

    data["weight_val"] = np.ones(len(data["labels_val"]))

    test = pd.read_csv('test.csv')

    test = test.fillna(' ')
    test.columns = ['test_id', 'id1', 'id2', 'question1', 'question2']
    data["test"] = test
    return data

def Glove_Indexing():
    # Indexing Glove
    print('Indexing word vectors.')
    embeddings_index = {}
    with codecs.open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def Words_Embedding(word_index, embeddings_index):
    # Embeddings
    print('Preparing embedding matrix')

    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix
