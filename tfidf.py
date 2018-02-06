import numpy as np


###############################################################################
# TFIDF Features
###############################################################################

def tfidf_features(df_features, tfidf):
    """
    Add new features to the dataframe :
    - Tf-Idf features
    :param df_features: dataframe
    :param tfidf: scikit learn trained tfidf object
    :return: updated dataframe
    """
    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))

    return df_features.fillna(0.0)
