from __future__ import division, print_function
import numpy as np
import pandas as pd
# Our features
import feature_engineering

################################################################################
############################ Feature Engineering ###############################
################################################################################

initial_train, initial_test = feature_engineering.create_text_and_graph_features()
tfidf_train, tfidf_test = feature_engineering.create_tfidf_features()
dist_train, dist_test = feature_engineering.create_distance_features()
pr_train, pr_test = feature_engineering.create_pagerank_features()

X_train = pd.concat([initial_train, tfidf_train, dist_train, pr_train], axis=1)
X_test = pd.concat([initial_test, tfidf_test, dist_test, pr_test], axis=1)

# save final features X_train and X_test to be re-used for the LSTM pre-processing
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
