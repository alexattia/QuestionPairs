import pandas as pd
import numpy as np

################################################################################
######################### Averaging XGBoost and LSTM ###########################
################################################################################

sub1 = pd.read_csv('lstm_final.csv', index_col=0)
sub2 = pd.read_csv('xgb_final.csv')
sub2["Lstm"] = sub1["score"]
sub2["Lstm2"] = sub1["score"]

# Averaging using best confidence
# def best_prob(row):
#     agm = np.argmax(np.abs(row - 0.5))
#     return row.loc[agm]
# dup = sub2.apply(best_prob, axis = 1)

dup = sub2.mean(axis = 1)
dup = pd.DataFrame(dup)
dup = dup.reset_index()
dup.columns = ['Id', 'Score']
dup.to_csv('submission.csv', index=False)
