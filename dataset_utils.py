import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
# text preprocessing
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


####################################################
### Load data
####################################################

def load_dataset():
    """
    Open train and test datasets and rename the columns
    :return: pandas dataframe train_set, test_set
    """
    df_train = pd.read_csv('train.csv', header=None)
    df_train = df_train.fillna(' ')
    df_train.columns = ['id', 'id1', 'id2', 'question1', 'question2', 'is_duplicate']
    df_test = pd.read_csv('test.csv', header=None)
    df_test.columns = ['id', 'id1', 'id2', 'question1', 'question2']
    return df_train, df_test


####################################################
### Clean data
####################################################

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

def clean_text(text, remove_stopwords=True, stem_words=False):
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
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can\'t", "cannot ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    #TODO CHIFFRES EN LETTRES (WORD2VEC PRIS EN COMPTE?)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return(text)
