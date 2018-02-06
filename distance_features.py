#from pygoose import *
import os
import warnings
import gensim
from fuzzywuzzy import fuzz
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import numpy as np
from dataset_utils import load_dataset

################################################################################
# All distance features
################################################################################

google_news_model_path = './GoogleNews-vectors-negative300.bin.gz'

def wmd(model, s1, s2):
    """
        Word Mover Distance
    """
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def norm_wmd(model, s1, s2):
    """
        Normalized Word Mover Distance
    """
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def sent2vec(model, s):
    """
    Sentence to Vector by summing the embeddings
    """
    words = s.lower()
    words = word_tokenize(words)
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def extend_with_features(data):
    stop_words = stopwords.words('english')
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

    model = gensim.models.KeyedVectors.load_word2vec_format(google_news_model_path, binary=True)
    data['wmd'] = data.apply(lambda x: wmd(model, x['question1'], x['question2']), axis=1)

    norm_model = gensim.models.KeyedVectors.load_word2vec_format(google_news_model_path, binary=True)
    norm_model.init_sims(replace=True)
    data['norm_wmd'] = data.apply(lambda x: norm_wmd(norm_model, x['question1'], x['question2']), axis=1)

    question1_vectors = np.zeros((data.shape[0], 300))
    for i, q in enumerate(data.question1.values):
        question1_vectors[i, :] = sent2vec(model, q)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in enumerate(data.question2.values):
        question2_vectors[i, :] = sent2vec(model, q)

    question1_vectors = np.nan_to_num(question1_vectors)
    question2_vectors = np.nan_to_num(question2_vectors)

    data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(question1_vectors, question2_vectors)]
    data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]

    data['skew_q1vec'] = [skew(x) for x in question1_vectors]
    data['skew_q2vec'] = [skew(x) for x in question2_vectors]
    data['kur_q1vec'] = [kurtosis(x) for x in question1_vectors]
    data['kur_q2vec'] = [kurtosis(x) for x in question2_vectors]
    return data
