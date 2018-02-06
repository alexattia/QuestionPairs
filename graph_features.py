import pandas as pd
from collections import defaultdict, Counter

def q1_freq(row, q_dict):
    """
    Compute the number of paired questions for the first question
    :param row: dataframe row
    :return: frequency of appareance 
    """
    return(len(q_dict[row['question1']]))

def q2_freq(row, q_dict):
    """
    Compute the number of paired questions for the second question
    :param row: dataframe row
    :return: frequency of appareance 
    """
    return(len(q_dict[row['question2']]))

def q1_q2_intersect(row, q_dict):
    """
    Compute the intersection of the number of paired questions for both questions (common neighbors of both questions)
    :param row: dataframe row
    :return: intersection of the frequencies of appearance 
    """
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


def get_graph_features(df_train, df_test):    
    ####################################################
    ### Create a dictionnary of paired questions => Graph struture
    ####################################################
    all_paired_ques = pd.concat([df_train[['question1', 'question2']], \
        df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(all_paired_ques.shape[0]):
            q_dict[all_paired_ques.question1[i]].add(all_paired_ques.question2[i])
            q_dict[all_paired_ques.question2[i]].add(all_paired_ques.question1[i])

    ####################################################        
    ### Graph features (frequency)
    ####################################################
    df_train['q1_q2_intersect'] = df_train.apply(lambda x: q1_q2_intersect(x, q_dict), axis=1, raw=True)
    df_train['q1_freq'] = df_train.apply(lambda x: q1_freq(x, q_dict), axis=1, raw=True)
    df_train['q2_freq'] = df_train.apply(lambda x: q2_freq(x, q_dict), axis=1, raw=True)

    df_test['q1_q2_intersect'] = df_test.apply(lambda x: q1_q2_intersect(x, q_dict), axis=1, raw=True)
    df_test['q1_freq'] = df_test.apply(lambda x: q1_freq(x, q_dict), axis=1, raw=True)
    df_test['q2_freq'] = df_test.apply(lambda x: q2_freq(x, q_dict), axis=1, raw=True)

    return df_train, df_test