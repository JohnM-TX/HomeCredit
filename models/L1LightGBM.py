
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



############################
#### DATA PREPROCESSING ####
############################

#%% treat missing values and proxies
traintest['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

traintest['TT_NULLCOUNT'] = traintest.isnull().sum(axis=1)


#%% treat outliers
traintest['AMT_INCOME_TOTAL'].clip_upper(200000, inplace=True)
traintest['AMT_CREDIT'].clip_upper(250000, inplace=True)
traintest['AMT_ANNUITY'].clip_upper(200000, inplace=True)
traintest['AMT_GOODS_PRICE'].clip_upper(200000, inplace=True)

traintest = traintest[traintest.CODE_GENDER != 'XNA']
traintest = traintest[traintest.NAME_INCOME_TYPE != 'Maternity leave']
traintest = traintest[traintest.NAME_FAMILY_STATUS != 'Unknown']




############################
#### FEATURE GENERATION ####
############################

# combine categories
traintest['CATCOMB1'] = traintest['FLAG_OWN_REALTY'] * trantest['NAME_HOUSING_TYPE']
traintest['CATCOMB2']  traintest['CODE_GENDER'], traintest['OCCUPATION_TYPE']
traintest['CATCOMB3']  traintest['CODE_GENDER'], traintest['FLAG_OWN_REALTY']

#%% encode highly cardinal categories
import category_encoders as ce
X = traintest['ORGANIZATION_TYPE'])
y = traintest['TARGET']

enc1 = ce.TargetEncoder(verbose=1)
encs1 = enc1.fit_transform(X, y)
all = all.join(encs1, rsuffix='_enc1')

enc2 = ce.LeaveOneOutEncoder(verbose=1)
encs2 = enc2.fit_transform(X,y)
all = all.join(encs2, rsuffix='_enc2')

#%% combine key numericals
traintest['ext_sources_mean'] = traintest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
traintest['annuity_income_percentage'] = traintest['AMT_ANNUITY'] / traintest['AMT_INCOME_TOTAL']
traintest['car_to_birth_ratio'] = traintest['OWN_CAR_AGE'] / traintest['DAYS_BIRTH']
traintest['car_to_employ_ratio'] = traintest['OWN_CAR_AGE'] / traintest['DAYS_EMPLOYED']
traintest['children_ratio'] = traintest['CNT_CHILDREN'] / traintest['CNT_FAM_MEMBERS']
traintest['credit_to_annuity_ratio'] = traintest['AMT_CREDIT'] / traintest['AMT_ANNUITY']
traintest['credit_to_goods_ratio'] = traintest['AMT_CREDIT'] / traintest['AMT_GOODS_PRICE']
traintest['credit_to_income_ratio'] = traintest['AMT_CREDIT'] / traintest['AMT_INCOME_TOTAL']
traintest['days_employed_percentage'] = traintest['DAYS_EMPLOYED'] / traintest['DAYS_BIRTH']
traintest['income_credit_percentage'] = traintest['AMT_INCOME_TOTAL'] / traintest['AMT_CREDIT']
traintest['income_per_child'] = traintest['AMT_INCOME_TOTAL'] / (1 + traintest['CNT_CHILDREN'])
traintest['income_per_person'] = traintest['AMT_INCOME_TOTAL'] / traintest['CNT_FAM_MEMBERS']
traintest['payment_rate'] = traintest['AMT_ANNUITY'] / traintest['AMT_CREDIT']
traintest['phone_to_birth_ratio'] = traintest['DAYS_LAST_PHONE_CHANGE'] / traintest['DAYS_BIRTH']
traintest['phone_to_employ_ratio'] = traintest['DAYS_LAST_PHONE_CHANGE'] / traintest['DAYS_EMPLOYED']

# bin numbers by groups
traintest['ANNUITY_GROUPED'] = traintest.groupby(['OCCUPATION_TYPE'])['AMT_ANNUITY'].transform('mean')
traintest.head().T

NUMERICAL_COLUMNS = ['AMT_ANNUITY',
                     'AMT_CREDIT',
                     'AMT_GOODS_PRICE',
                     'AMT_INCOME_TOTAL',
                     'AMT_REQ_CREDIT_BUREAU_YEAR',
                     'DAYS_BIRTH',
                     'EXT_SOURCE_1', 
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3']

CAT_COLUMNS =  [['OCCUPATION_TYPE'],
                ['CODE_GENDER', 'NAME_EDUCATION_TYPE']
                ['FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE'],
                ['CODE_GENDER', 'ORGANIZATION_TYPE']]

for agg in ['mean', 'max', 'sum']:
    for numcol in NUMERICAL_COLUMNS:
        for catgroup in CAT_COLUMNS:
            traintest[numcol+'_'.join(catgroup)+agg] = traintest.groupby(catgroup)[numcol].transform(agg)

# more bins and flags
traintest['long_employment'] = (traintest['DAYS_EMPLOYED'] > -2000).astype(int)
traintest['retirement_age'] = (traintest['DAYS_BIRTH'] > -14000).astype(int)


######  BREAK AND RUN MODEL #####




#%% add in previous applications
prev = pd.read_csv('./input/raw/previous_application.csv', index_col='SK_ID_PREV')
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True).head()

prev.replace(365243, np.nan, inplace=True)
prev['PA_NULLCOUNT'] = prev.isnull().sum(axis=1)


PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]




prev_applications_sortedprev_ap ['previous_application_prev_was_refused'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_refused'].last().reset_index()

    prev_applications_sorted['previous_application_prev_was_approved'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_approved'].last().reset_index()





# merge



#################
#### MODELING ####
##################


 lgbm__boosting_type: gbdt
  lgbm__objective: binary
  lgbm__metric: auc
  lgbm__number_boosting_rounds: 5000
  lgbm__early_stopping_rounds: 100
  lgbm__learning_rate: 0.1
  lgbm__max_bin: 300
  lgbm__max_depth: -1
  lgbm__num_leaves: 35
  lgbm__min_child_samples: 50
  lgbm__subsample: 1.0
  lgbm__subsample_freq: 1
  lgbm__colsample_bytree: 0.2
  lgbm__min_gain_to_split: 0.5
  lgbm__reg_lambda: 100.0
  lgbm__reg_alpha: 0.0
  lgbm__scale_pos_weight: 1