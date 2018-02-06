from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


## Load Train/Test features
X_train = pd.read_csv("X_train.csv", index_col=0)
X_test = pd.read_csv("X_test.csv", index_col=0)

y_train = X_train.is_duplicate
X_train = X_train.drop(['id', 'id1', 'id2', "question1", "question2", 'is_duplicate'],
                      axis=1)
X_test = X_test.drop(['id', 'id1', 'id2', "question1", "question2"],
                      axis=1)
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0.0)
X_test = X_test.fillna(0.0)
# Train Val Splitting
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


################################################################################
############################ Random Forest #####################################
################################################################################


clf = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

with open("submission_file.csv", 'w') as f:
    f.write("Id,Score\n")
    for i in range(y_pred.shape[0]):
        f.write(str(i)+','+str(y_pred[i][1])+'\n')
