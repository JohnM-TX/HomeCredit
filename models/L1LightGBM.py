
#%% get libraries
import os
import gc
import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# set the level
Level2 = True  # adds full pipeline
Level3 = False # adds bureau data


#%%################
#### LOAD DATA ####
###################
if True:
    train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
    test = pd.read_csv('./input/raw/application_test.csv', index_col='SK_ID_CURR')
    test['TARGET'] = 2
    traintest = pd.concat([train, test], sort=False).sort_index()

traintest.head().T


#%%######################
#### PREPROCESS DATA ####
#########################
if Level2: 
    # treat outliers and missings
    numcols = test.select_dtypes(exclude='object').columns.tolist()
    numcols.remove('TARGET')
    for n in numcols:
        min_ = np.amin(test[n])
        max_ = np.amax(test[n])
        train[n] = train[n].clip(min_, max_)
        
    train = train[train.CODE_GENDER != 'XNA']
    train = train[train.NAME_INCOME_TYPE != 'Maternity leave']
    train = train[train.NAME_FAMILY_STATUS != 'Unknown']
    
    traintest = pd.concat([train, test], sort=False).sort_index()
    traintest['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    traintest['TT_NULLCOUNT'] = traintest.isnull().sum(axis=1)

del train; del test
gc.collect()
traintest.head().T


#%%########################
#### GENERATE FEATURES ####
###########################
if Level2: 
    # combine categories
    traintest['HOME_OWNER_TYPE'] = traintest['FLAG_OWN_REALTY'] + traintest['NAME_HOUSING_TYPE']
    traintest['OCC_TYPE_GENDER'] = traintest['CODE_GENDER'] + traintest['OCCUPATION_TYPE']
    traintest['HOME_OWNER_GENDER'] = traintest['CODE_GENDER'] + traintest['FLAG_OWN_REALTY']

    # encode highly cardinal categories
    traintest['ORGANIZATION_TYPE'].nunique()
    X = traintest['ORGANIZATION_TYPE'].values
    y = traintest['TARGET'].values

    enc1 = ce.TargetEncoder(verbose=1)
    encs1 = enc1.fit_transform(X, y)
    traintest = traintest.join(encs1, rsuffix='_enc1')
    del encs1

    # combine key numericals
    traintest['ext_sources_mean'] = traintest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    traintest['ext_sources_std'] = traintest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)

    traintest['annuity_income_percentage'] = traintest['AMT_ANNUITY'] / traintest['AMT_INCOME_TOTAL']
    traintest['car_to_birth_ratio'] = traintest['OWN_CAR_AGE'] / traintest['DAYS_BIRTH']
    traintest['car_to_employ_ratio'] = traintest['OWN_CAR_AGE'] / traintest['DAYS_EMPLOYED']
    traintest['children_ratio'] = traintest['CNT_CHILDREN'] / traintest['CNT_FAM_MEMBERS']
    traintest['credit_to_annuity_ratio'] = traintest['AMT_CREDIT'] / traintest['AMT_ANNUITY']
    traintest['credit_to_goods_ratio'] = traintest['AMT_CREDIT'] / traintest['AMT_GOODS_PRICE']
    traintest['days_employed_percentage'] = traintest['DAYS_EMPLOYED'] / traintest['DAYS_BIRTH']
    traintest['income_per_child'] = traintest['AMT_INCOME_TOTAL'] / (1 + traintest['CNT_CHILDREN'])
    traintest['income_per_person'] = traintest['AMT_INCOME_TOTAL'] / traintest['CNT_FAM_MEMBERS']
    traintest['payment_rate'] = traintest['AMT_ANNUITY'] / traintest['AMT_CREDIT']
    traintest['phone_to_birth_ratio'] = traintest['DAYS_LAST_PHONE_CHANGE'] / traintest['DAYS_BIRTH']
    traintest['phone_to_employ_ratio'] = traintest['DAYS_LAST_PHONE_CHANGE'] / traintest['DAYS_EMPLOYED']

    #%% # bin numbers by groups
    traintest['ANNUITY_GROUPED'] = traintest.groupby(['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'])['EXT_SOURCE_1'].transform('mean')
    

    NUMERICAL_COLUMNS = ['AMT_ANNUITY',
                        'AMT_CREDIT',
                        'AMT_INCOME_TOTAL',
                        'AMT_REQ_CREDIT_BUREAU_YEAR',
                        'DAYS_BIRTH',
                        'EXT_SOURCE_1', 
                        'EXT_SOURCE_2',
                        'EXT_SOURCE_3',
                        'AMT_REQ_CREDIT_BUREAU_YEAR'
                        ]

    CAT_COLUMNS =  [['OCCUPATION_TYPE'],
                    ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
                    ['FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE'],
                    ['CODE_GENDER', 'ORGANIZATION_TYPE'],
                    ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
                    ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY']]

    for agg in ['mean', 'max', 'min']:
        for numcol in NUMERICAL_COLUMNS:
            for catgroup in CAT_COLUMNS:
                traintest[numcol+'_'.join(catgroup)+agg] = traintest.groupby(catgroup)[numcol].transform(agg)

    #%% more bins and flags
    traintest['previous_employment'] = (traintest['DAYS_EMPLOYED'] > -2000).astype(int)
    traintest['retirement_age'] = (traintest['DAYS_BIRTH'] > -14000).astype(int)


#%%#######################
#### SELECT FEATURES #####
##########################
if Level2: 
    #%% drop columns of no use
    uselesses = ['FLAG_DOCUMENT_10',
                        'FLAG_DOCUMENT_12',
                        'FLAG_DOCUMENT_13',
                        'FLAG_DOCUMENT_14',
                        'FLAG_DOCUMENT_15',
                        'FLAG_DOCUMENT_16',
                        'FLAG_DOCUMENT_17',
                        'FLAG_DOCUMENT_19',
                        'FLAG_DOCUMENT_2',
                        'FLAG_DOCUMENT_20',
                        'FLAG_DOCUMENT_21',
                        'FLAG_DOCUMENT_4',
                        'FLAG_DOCUMENT_7',
                        'FLAG_DOCUMENT_9']


    traintest = traintest.drop(uselesses, axis=1)

traintest.head().T
traintest.shape


#%%##############################
#### ENGINEER MORE FEATURES #####
#################################
if (Level2 and Level3):

    # Make function for one-hot (DISABLED)
    def one_hot_encoder(df, nan_as_category = True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns


    # Preprocess bureau.csv and bureau_balance.csv
    def bureau_and_balance(num_rows = None, nan_as_category = True):
        bureau = pd.read_csv('./input/raw/bureau.csv', nrows = num_rows)
        bb = pd.read_csv('./input/raw/bureau_balance.csv', nrows = num_rows)
        bb, bb_cat = one_hot_encoder(bb, nan_as_category)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
        
        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        del bb, bb_agg
        gc.collect()
        
        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': [ 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': [ 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': [ 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
        
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        return bureau_agg

    # Preprocess previous_applications.csv
    def previous_applications(num_rows = None, nan_as_category = True):
        prev = pd.read_csv('./input/raw/previous_application.csv', nrows = num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': [ 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']
        
        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg

    # Preprocess POS_CASH_balance.csv
    def pos_cash(num_rows = None, nan_as_category = True):
        pos = pd.read_csv('./input/raw/POS_CASH_balance.csv', nrows = num_rows)
        pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
        
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        
        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        del pos
        gc.collect()
        return pos_agg
        
    # Preprocess installments_payments.csv
    def installments_payments(num_rows = None, nan_as_category = True):
        ins = pd.read_csv('./input/raw/installments_payments.csv', nrows = num_rows)
        ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
        
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': [ 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': [ 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        gc.collect()
        return ins_agg

    # Preprocess credit_card_balance.csv
    def credit_card_balance(num_rows = None, nan_as_category = True):
        cc = pd.read_csv('./input/raw/credit_card_balance.csv', nrows = num_rows)
        cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
        
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg

    # run functions and merge one at a time to save memory
    bureau = bureau_and_balance()
    traintest = traintest.join(bureau)
    del bureau
    print('bureau segment done')

    prev = previous_applications()
    traintest = traintest.join(prev)
    del prev
    print('prev segment done')

    pos = pos_cash()
    traintest = traintest.join(pos)
    del pos
    print('pos segment done')

    ins = installments_payments()
    traintest = traintest.join(ins)
    del ins
    print('ins segment done')

    cc = credit_card_balance()
    traintest = traintest.join(cc)
    del cc
    print('cc segment done')

traintest.shape


#%%#############
#### MODEL ####
###############
# prep for model (all rounds)
if True:
    objcols = traintest.select_dtypes('object')
    for c in objcols.columns:
        traintest[c] = traintest[c].astype('str')
        traintest[c] = LabelEncoder().fit_transform(traintest[c])
        if traintest[c].nunique() < 50:
            traintest[c] = traintest[c].astype('category')

    train = traintest[traintest.TARGET != 2]
    test = traintest[traintest.TARGET == 2]

    train.dtypes[train.dtypes=='category'] ####
    train.select_dtypes('category').dtypes ####
    print(train.shape, test.shape)

    X = train.drop('TARGET', axis=1)
    y = train['TARGET']
    X_test = test.drop('TARGET', axis=1)

    # del(traintest)
    gc.collect()

train.head().T


#%% split data and run model 
if not Level2: # Level1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32, 
        stratify=y)

    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=35, max_depth=-1, 
        learning_rate=0.02, n_estimators=1000, subsample_for_bin=200000, objective='binary', 
        class_weight=None, min_child_samples=40, subsample=1.0, reg_lambda=50.0, 
        predict_contrib=True, subsample_freq=0, colsample_bytree=0.8, silent=False) 

    lmod.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
        early_stopping_rounds=50, verbose=20)

    sub_preds = lmod.predict_proba(X_test)[:, 1]


elif not Level3: # Level2
    # add more parameters and use k-fold
    params = {'objective':'binary',
              'metric':'auc',
              'boosting':'gbdt', 
              'num_leaves':40, 
              'max_depth':6, 
              'learning_rate':0.02, 
              'subsample_for_bin':200000, 
              'class_weight':None, 
              'min_child_samples':40, 
              'subsample':0.9, 
              'reg_lambda':50.0, 
              'predict_contrib':True, 
              'subsample_freq':0, 
              'colsample_bytree':0.85,
              'num_threads':3}
    sub_preds = np.zeros_like(test.TARGET, dtype=float)
    oof_preds = np.zeros_like(y, dtype=float)
    cv = StratifiedKFold(n_splits=4, random_state=42)
    for trn_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
        trainDS = lgb.Dataset(X_train, label=y_train.values)
        valDS = lgb.Dataset(X_val, label=y_val.values, reference=trainDS)

        evalnums = {}
        lmod = lgb.train(params, trainDS, num_boost_round=1400, early_stopping_rounds=50,
            valid_sets=[trainDS, valDS], evals_result=evalnums, verbose_eval=20)  # 1400 rds

        oof_preds[val_idx] = lmod.predict(X_val)
        sub_preds += lmod.predict(X_test.values)/cv.n_splits
    
    roc_auc_score(y, oof_preds)


else: # Level3
    # adjust parameters to accomodate new features    
    
    params = {'objective':'binary',
            'metric':'auc',
            'boosting':'gbdt', 
            'num_leaves':40,  
            'max_depth':8,  #6
            'learning_rate':0.02, 
            'subsample_for_bin':200000, 
            'class_weight':None, 
            'min_child_samples':60, #40
            'subsample':0.9, 
            'reg_lambda':10.0,  #10
            'reg_alpha':10.0,   ###
            'min_data_in_leaf':1000,    ###
            'predict_contrib':True, 
            'subsample_freq':0, 
            'colsample_bytree':0.75,  #0.85
            'num_threads':3}

    sub_preds = np.zeros_like(test.TARGET, dtype=float)
    oof_preds = np.zeros_like(y, dtype=float)
    cv = StratifiedKFold(n_splits=4, random_state=42)
    for trn_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
        trainDS = lgb.Dataset(X_train, label=y_train.values)
        valDS = lgb.Dataset(X_val, label=y_val.values, reference=trainDS)
        
        evalnums = {}
        lmod = lgb.train(params, trainDS, num_boost_round=1400, early_stopping_rounds=50,
            valid_sets=[trainDS, valDS], evals_result=evalnums, verbose_eval=20)
        
        oof_preds[val_idx] = lmod.predict(X_val)
        sub_preds += lmod.predict(X_test.values)/cv.n_splits
     
    roc_auc_score(y, oof_preds)   


#%%#############
#### SUBMIT ####
################
# predict test targets
sub = pd.read_csv('./input/raw/sample_submission.csv', index_col='SK_ID_CURR')
sub['TARGET'] = np.around(sub_preds, 4)
sub.head()
sub.to_csv('./subs/sub_test.csv')



#%%#############
#### REVIEW ####
################
# get model info
lmod.best_iteration_
lmod.best_score_.get('valid_0')

# show learning curve (python API)
ax = lgb.plot_metric(evalnums, metric='auc')
plt.show()

# plot gains
lgb.plot_importance(lmod, importance_type='gain', max_num_features=-1, figsize=(4,50))

# plot model tree
graph = lgb.create_tree_digraph(lmod, 1, show_info=['split_gain', 'leaf_count'], 
    format='pdf')
graph.render(view=True)









# Round, CV, LB
# rd1, 0.761, 0.746
# rd2,  