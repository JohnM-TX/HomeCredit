
#%% get libraries
import os
import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100


#%% get data and peek
train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
test = pd.read_csv('./input/raw/application_test.csv', index_col='SK_ID_CURR')
test['TARGET'] = 2
traintest = pd.concat([train, test], sort=False).sort_index()
traintest.head().T


#%% treat missing values





#%% treat outliers
np.clip or treat as missing




#%% Cat feature gen

Frequency Encoding
Mean Target Encoding for high cats
Cat combos


#%% Num feature gen
Num combos





