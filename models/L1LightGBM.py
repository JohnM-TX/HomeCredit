
#%% get libraries
import os
import gc
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


#%%
# Level2 adds helpful steps
# Level3 adds bureau data 

Level2 = True
Level3 = True

if Level2: 

    ############################
    #### DATA PREPROCESSING ####
    ############################

    #%% treat missing values and proxies
    # traintest['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    traintest['TT_NULLCOUNT'] = traintest.isnull().sum(axis=1)


    #BAD
    #%% treat outliers
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
    traintest.head().T

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
                        'FLAG_DOCUMENT_21',
                        'FLAG_DOCUMENT_4',
                        'FLAG_DOCUMENT_7',
                        'FLAG_DOCUMENT_9']


    traintest = traintest.drop(USELESS_COLUMNS, axis=1)

traintest.shape




###########################################
#%% for round 3

if (Level2 and Level3):

    ############################
    #### FEATURE SELECTION #####
    ############################

    # Preprocess bureau.csv and bureau_balance.csv
    def one_hot_encoder(df, nan_as_category = True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns


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


    bureau = bureau_and_balance()
    traintest = traintest.join(bureau)
    # del bureau
    gc.collect()
    print('segment done')

    prev = previous_applications()
    traintest = traintest.join(prev)
    # del prev
    gc.collect()
    print('segment done')

    pos = pos_cash()
    traintest = traintest.join(pos)
    # del pos
    gc.collect()
    print('segment done')

    ins = installments_payments()
    traintest = traintest.join(ins)
    # del ins
    gc.collect()
    print('segment done')

    cc = credit_card_balance()
    traintest = traintest.join(cc)
    # del cc
    gc.collect()
    print('segment done')

    traintest.sort_index(inplace=True)
    
traintest.head(10).T



#################
#### MODELING ####
##################

#%% prep for model

catcols = []
for c in traintest.columns:
    if traintest[c].dtype == 'object':
        catcols.append(c)
        traintest[c] = traintest[c].astype('category').cat.codes
catcols

train = traintest[traintest.TARGET != 2]
train.shape

#%%
# split data and run model (you should really use stratified k-fold here)
X = train.drop('TARGET', axis=1)
y = train.TARGET
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if not Level2:
    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=35, max_depth=-1, learning_rate=0.02, 
        n_estimators=1000, subsample_for_bin=200000, objective='binary', class_weight=None, 
        min_child_samples=40, subsample=1.0, reg_lambda=50.0,
        subsample_freq=0, colsample_bytree=0.8, silent=False) 

elif not Level3:
    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=6, learning_rate=0.02, 
        n_estimators=1200, subsample_for_bin=200000, objective='binary', class_weight=None, 
        min_child_samples=40, subsample=0.9, reg_lambda=50.0,
        subsample_freq=0, colsample_bytree=0.85, silent=False) 

else:
    lmod = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=8, learning_rate=0.02, 
        n_estimators=1400, subsample_for_bin=200000, objective='binary', class_weight=None, 
        min_child_samples=60, subsample=0.9, reg_lambda=10.0, reg_alpha=1.0, min_data_in_leaf=1000,
        subsample_freq=0, colsample_bytree=0.75, silent=False) 


lmod.fit(X_train, y_train, eval_set=[(X_val, y_val)],  eval_metric='auc', 
    early_stopping_rounds=50, verbose=True)

lmod.best_score_.get('valid_0')

#%% get model info
print(lmod.best_score_.get('valid_0'))
featmat = pd.DataFrame({'feat':X.columns, 'imp':lmod.feature_importances_})
featmat.sort_values('imp', ascending=False)
drops = featmat.loc[featmat.imp == 0, 'feat'].tolist()

#%% plot learning curves and tree structure
lgb.create_tree_digraph(lmod, filename='rest.gv')


train.drop(drops, axis=1, inplace=True)
train.shape