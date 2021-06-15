#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score 
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve ,auc , log_loss ,  classification_report 
from sklearn.model_selection import train_test_split
from IPython import embed

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    #return 'f1', f1_score(y_true, y_hat), True
    return 'f2', fbeta_score(y_true, y_hat, beta=5), True

df_X = pd.read_csv("./train_new.csv")
df_X.replace('N', 0, inplace=True)
df_X.replace('Y', 1, inplace=True)
df_X = df_X.set_index("txkey")
#df_y = pd.read_csv("./test.csv")

df_X_train, df_X_val = train_test_split(df_X, test_size=0.25, random_state=42)
df_Y_train = df_X_train['fraud_ind']
df_Y_val = df_X_val['fraud_ind']
df_X_train.drop('fraud_ind', inplace=True, axis=1)
df_X_val.drop('fraud_ind', inplace=True, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(df_X_train, df_Y_train)
lgb_eval = lgb.Dataset(df_X_val, df_Y_val, reference=lgb_train)

# specify your configurations as a dict
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='binary' #Binary target feature
#params['metric']='binary_logloss' #metric for binary classification
params['metric']='None'
params['max_depth']=10

print('Starting training...')

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=lgb_eval,
                early_stopping_rounds=400,
                feval=lgb_f1_score)

df_X_test = pd.read_csv("./test_new.csv")
df_X_test.replace('N', 0, inplace=True)
df_X_test.replace('Y', 1, inplace=True)
df_X_test = df_X_test.set_index("txkey")
df_Y_test = df_X_test['fraud_ind']
df_X_test.drop('fraud_ind', inplace=True, axis=1)

y_pred = gbm.predict(df_X_test, num_iteration=gbm.best_iteration)

print('precision:', precision_score(df_Y_test, y_pred>0.5))
print('recall:', recall_score(df_Y_test, y_pred>0.5))
print('f1:', f1_score(df_Y_test, y_pred>0.5))
#y_test.mean()
TP, TN, FP, FN = 0, 0, 0, 0
for p, a in zip(y_pred, df_Y_test):
    if p < 0.5 and a < 0.5:
        TN += 1
    elif p >= 0.5 and a >= 0.5:
        TP += 1
    elif p >= 0.5 and a < 0.5:
        FP += 1
    elif p < 0.5 and a >= 0.5:
        FN += 1
    else:
        print('jizz')

embed()

