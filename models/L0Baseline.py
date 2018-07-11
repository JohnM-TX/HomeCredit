
#%% get libraries
import os
import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

#%% get data and peek
train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
test = pd.read_csv('./input/raw/application_test.csv', index_col='SK_ID_CURR')
test['TARGET'] = 2
traintest = pd.concat([train, test], sort=False).sort_index()
traintest.head()

#%% prep for model
for c in traintest.columns:
    if traintest[c].dtype == 'object':
        traintest[c] = traintest[c].astype('category').cat.codes

train = traintest[traintest.TARGET != 2]
test = traintest[traintest.TARGET == 2]


#%% split data and run model
X = train.drop('TARGET', axis=1)
y = train.TARGET
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=4, learning_rate=0.05, 
    n_estimators=1000, subsample_for_bin=200000, objective='binary', class_weight=None, 
    min_child_samples=10, subsample=1.0, 
    subsample_freq=0, colsample_bytree=0.9, silent=False) 

lmod.fit(X_train, y_train, eval_set=[(X_val, y_val)],  eval_metric='auc', 
    early_stopping_rounds=50, verbose=True)

#%% get info
print(lmod.best_score_)
featmat = pd.DataFrame({'feat':X.columns, 'imp':lmod.feature_importances_})
featmat.sort_values('imp', ascending=False)

#%%
preds = lmod.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, preds))
evalmat = pd.DataFrame({'labels':y_val, 'preds':preds})
evalmat.sort_values('preds').head(10)
evalmat.sort_values('preds').tail(10)
