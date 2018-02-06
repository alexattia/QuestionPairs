from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
# XGBoost library
import xgboost as xgb
from xgboost import XGBClassifier

X_train = pd.read_csv("X_train.csv", index_col=0)
X_test = pd.read_csv("X_test.csv", index_col=0)

y_train = X_train.is_duplicate
X_train = X_train.drop(['id', 'id1', 'id2', "question1", "question2", 'is_duplicate'],
                      axis=1)
X_test = X_test.drop(['id', 'id1', 'id2', "question1", "question2"],
                      axis=1)

# Train Val Splitting
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


################################################################################
############################ Model Tuning ######################################
################################################################################

# Multi Xgboost
df_tot = pd.DataFrame()
for i in range(9):
    np.random.seed(i+1)
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    et = [.02,.025,.01,.015]
    params['eta'] = np.random.choice(et)
    params['n_jobs'] = 5
    depth = [4,5,6,7]
    params['max_depth'] = np.random.choice(depth)
    sub = [.5,.6,.7,.4]
    params['subsample'] = np.random.choice(sub)
    params['base_score'] = 0.2
    col = [1,.7]
    params['colsample_bytree'] = np.random.choice(col)

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=2500)

    d_test = xgb.DMatrix(X_test)
    p_test = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)
    df_tot[str('fold'+ str(i+1))] = p_test

# Save results from each XGBoost
df_tot.to_csv('xgb_final.csv', index=False)
