
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
traintest.head().T


# bare = True will go straight into modeling
bare=False

if not bare: 
   
    
    ############################
    #### DATA PREPROCESSING ####
    ############################
    

   
    #%% treat missing values and proxies
    # traintest['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    traintest['TT_NULLCOUNT'] = traintest.isnull().sum(axis=1)

    #BAD
    # #%% treat outliers
    # traintest['AMT_INCOME_TOTAL'].clip_upper(200000, inplace=True)
    # traintest['AMT_CREDIT'].clip_upper(250000, inplace=True)
    # traintest['AMT_ANNUITY'].clip_upper(200000, inplace=True)
    # traintest['AMT_GOODS_PRICE'].clip_upper(200000, inplace=True)

    traintest = traintest[traintest.CODE_GENDER != 'XNA']
    traintest = traintest[traintest.NAME_INCOME_TYPE != 'Maternity leave']
    traintest = traintest[traintest.NAME_FAMILY_STATUS != 'Unknown']




    ############################
    #### FEATURE GENERATION ####
    ############################

    #%% combine categories
    traintest['CATCOMB1'] = traintest['FLAG_OWN_REALTY'] + traintest['NAME_HOUSING_TYPE']
    traintest['CATCOMB2'] = traintest['CODE_GENDER'] + traintest['OCCUPATION_TYPE']
    traintest['CATCOMB3'] = traintest['CODE_GENDER'] + traintest['FLAG_OWN_REALTY']

    # encode highly cardinal categories
    import category_encoders as ce
    X = traintest['ORGANIZATION_TYPE'].values
    y = traintest['TARGET'].values

    enc1 = ce.TargetEncoder(verbose=1)
    encs1 = enc1.fit_transform(X, y)
    traintest = traintest.join(encs1, rsuffix='_enc1')


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

    #%% # bin numbers by groups
    traintest['ANNUITY_GROUPED'] = traintest.groupby(['OCCUPATION_TYPE'])['AMT_ANNUITY'].transform('mean')
    traintest.head().T

    # NUMERICAL_COLUMNS = ['AMT_ANNUITY',
    #                     'AMT_CREDIT',
    #                     'AMT_GOODS_PRICE',
    #                     'AMT_INCOME_TOTAL',
    #                     'AMT_REQ_CREDIT_BUREAU_YEAR',
    #                     'DAYS_BIRTH',
    #                     'EXT_SOURCE_1', 
    #                     'EXT_SOURCE_2',
    #                     'EXT_SOURCE_3']

    # CAT_COLUMNS =  [['OCCUPATION_TYPE'],
    #                 ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
    #                 ['FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE'],
    #                 ['CODE_GENDER', 'ORGANIZATION_TYPE']]

    # for agg in ['mean', 'max', 'sum']:
    #     for numcol in NUMERICAL_COLUMNS:
    #         for catgroup in CAT_COLUMNS:
    #             traintest[numcol+'_'.join(catgroup)+agg] = traintest.groupby(catgroup)[numcol].transform(agg)

    #%% more bins and flags
    traintest['long_employment'] = (traintest['DAYS_EMPLOYED'] > -2000).astype(int)
    traintest['retirement_age'] = (traintest['DAYS_BIRTH'] > -14000).astype(int)



    ############################
    #### FEATURE SELECTION #####
    ############################

    #%% drop columns of no use
    USELESS_COLUMNS = ['FLAG_DOCUMENT_10',
                    'FLAG_DOCUMENT_12',
                    'FLAG_DOCUMENT_13',
                    'FLAG_DOCUMENT_14',
                    'FLAG_DOCUMENT_15',
                    'FLAG_DOCUMENT_16',
                    'FLAG_DOCUMENT_17',
                    'FLAG_DOCUMENT_19',
                    'FLAG_DOCUMENT_2',
                    'FLAG_DOCUMENT_20',
                    'FLAG_DOCUMENT_21']


    traintest = traintest.drop(USELESS_COLUMNS, axis=1)

traintest.shape


#%% for round 3


    # #%% add in previous applications
    # prev = pd.read_csv('./input/raw/previous_application.csv', index_col='SK_ID_PREV')
    # prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True).head()

    # prev.replace(365243, np.nan, inplace=True)
    # prev['PA_NULLCOUNT'] = prev.isnull().sum(axis=1)


    # PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
    # for agg in ['mean', 'min', 'max', 'sum', 'var']:
    #     for select in ['AMT_ANNUITY',
    #                    'AMT_APPLICATION',
    #                    'AMT_CREDIT',
    #                    'AMT_DOWN_PAYMENT',
    #                    'AMT_GOODS_PRICE',
    #                    'CNT_PAYMENT',
    #                    'DAYS_DECISION',
    #                    'HOUR_APPR_PROCESS_START',
    #                    'RATE_DOWN_PAYMENT'
    #                    ]:
    #         PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
    # PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]


    # prev_applications_sortedprev_ap ['previous_application_prev_was_refused'] = (
    #         prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    # group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    #     'previous_application_prev_was_refused'].last().reset_index()

    #     prev_applications_sorted['previous_application_prev_was_approved'] = (
    #         prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    # group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    #     'previous_application_prev_was_approved'].last().reset_index()

    # merge



#################
#### MODELING ####
##################

#%% prep for model
traintest.dtypes

catcols = []
for c in traintest.columns:
    if traintest[c].dtype == 'object':
        catcols.append(c)
        traintest[c] = traintest[c].astype('category').cat.codes
catcols

train = traintest[traintest.TARGET != 2]
train.shape


# split data and run model
X = train.drop('TARGET', axis=1)
y = train.TARGET
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if bare:
    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=35, max_depth=-1, learning_rate=0.02, 
        n_estimators=1000, subsample_for_bin=200000, objective='binary', class_weight=None, 
        min_child_samples=40, subsample=1.0, reg_lambda=50.0,
        subsample_freq=0, colsample_bytree=0.8, silent=False) 

else:
    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, max_depth=8, learning_rate=0.022, 
        n_estimators=1000, subsample_for_bin=200000, objective='binary', class_weight=None, 
        min_child_samples=40, subsample=0.8, reg_lambda=50.0,
        subsample_freq=0, colsample_bytree=0.8, silent=False) 

lmod.fit(X_train, y_train, eval_set=[(X_val, y_val)],  eval_metric='auc', 
    early_stopping_rounds=50, verbose=True)

lmod.best_score_.get('valid_0')

#%% get info
print(lmod.best_score_.get('valid_0'))
featmat = pd.DataFrame({'feat':X.columns, 'imp':lmod.feature_importances_})
featmat.sort_values('imp', ascending=False)











# drops = featmat.loc[featmat.imp == 0, 'feat'].tolist()

# #%% drop non-predictors and rerun
# traintest1 = traintest.drop(drops, axis=1)

# train1 = traintest1[traintest.TARGET != 2]
# train1.shape

# #%%
# X = train1.drop('TARGET', axis=1)
# y = train1.TARGET
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=35, max_depth=-1, learning_rate=0.1, 
# #     n_estimators=2000, subsample_for_bin=200000, objective='binary', class_weight=None, 
# #     min_child_samples=50, subsample=1.0, reg_lambda=50.0,
# #     subsample_freq=0, colsample_bytree=0.75, silent=False) 


# lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=51, max_depth=-1, learning_rate=0.05, 
#     n_estimators=1000, subsample_for_bin=200000, objective='binary', class_weight=None, 
#     min_child_samples=10, subsample=1.0, 
#     subsample_freq=0, colsample_bytree=0.9, silent=False) 

# lmod.fit(X_train, y_train, eval_set=[(X_val, y_val)],  eval_metric='auc', 
#     early_stopping_rounds=60, verbose=True)

# print(lmod.best_score_)
