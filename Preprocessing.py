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

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "i am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" USA ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r" UK ", " england ", text)
    text = re.sub(r"india", "india", text)
    text = re.sub(r"kms", " kilometers ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

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
    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)
    return data_1, data_2, test_data_1, test_data_2, test_ids, labels, word_index

def load_leaky():
    # new features
    print("-- Preprocessing - load leaky features")
    trn = pd.read_csv('X_train.csv')
    trn = trn.drop(["question1", "question2",
                    "question1_nouns", "question2_nouns"],
                          axis=1)
    tst = pd.read_csv('X_test.csv')
    tst = tst.drop(["question1", "question2",
                            "question1_nouns", "question2_nouns"],
                          axis=1)
    trn = trn.fillna(value=0)
    tst = tst.fillna(value=0)
    trn.shape, tst.shape
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
