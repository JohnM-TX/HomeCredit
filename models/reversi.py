
# In[1]
import gc
import pandas as pd
import numpy as np


#%% get installments data
inst = pd.read_csv('./input/raw/installments_payments.csv')
inst.sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], inplace=True)

# aggregate split payments
_aggdict = {'SK_ID_CURR':'first', 
            'NUM_INSTALMENT_VERSION':'first',
            'DAYS_INSTALMENT':'mean',
            'DAYS_ENTRY_PAYMENT':'mean', 
            'AMT_INSTALMENT':'mean', 
            'AMT_PAYMENT':'sum'
            }
_grouped = inst.groupby(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 
    as_index=False) # sort=True
inst_agg0 = _grouped.agg(_aggdict)

inst_agg0['amtdiff'] = inst_agg0.AMT_INSTALMENT - inst_agg0.AMT_PAYMENT
inst_agg0['daydiff'] = inst_agg0.DAYS_INSTALMENT - inst_agg0.DAYS_ENTRY_PAYMENT
inst_agg0['ncount'] = inst_agg0.groupby(['SK_ID_PREV'])['SK_ID_CURR'].transform('count')
inst_agg0.head().T

# now aggregate to idprev
_aggdict = {'NUM_INSTALMENT_NUMBER':'max',
            'DAYS_INSTALMENT':'min',
            'DAYS_ENTRY_PAYMENT':'min', 
            'AMT_INSTALMENT':'std', 
            'AMT_PAYMENT':'std',
            'amtdiff':'sum',
            'daydiff':'sum'
            }
_grouped = inst_agg0.groupby(['SK_ID_PREV']) # sort=True
inst_agg = _grouped.agg(_aggdict)
inst_agg.head().T


#%% read revolving data
revbal = pd.read_csv('./input/raw/credit_card_balance.csv')
revbal.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
revbal['completed'] = np.where(revbal.NAME_CONTRACT_STATUS == 'Completed',
    1, 0)
revbal.head().T


# aggregate to id prev
_aggdict =    { 'MONTHS_BALANCE':'min',
                'AMT_BALANCE':'max',
                'AMT_CREDIT_LIMIT_ACTUAL':'last',
                'AMT_DRAWINGS_ATM_CURRENT':'last',
                'AMT_DRAWINGS_CURRENT':'last',
                'AMT_DRAWINGS_OTHER_CURRENT':'last',
                'AMT_DRAWINGS_POS_CURRENT':'last',
                'AMT_INST_MIN_REGULARITY':'last',
                'AMT_PAYMENT_CURRENT':'last',
                'AMT_PAYMENT_TOTAL_CURRENT':'last',
                'AMT_RECEIVABLE_PRINCIPAL':'last',
                'AMT_RECIVABLE':'last',
                'CNT_DRAWINGS_CURRENT':'mean',
                'CNT_DRAWINGS_OTHER_CURRENT':'mean',
                'CNT_DRAWINGS_POS_CURRENT':'mean',
                'completed': 'sum'
               }
_grouped = revbal.groupby(['SK_ID_PREV']) # sort=True
rev_agg = _grouped.agg(_aggdict)
rev_agg.head().T




#%% read cash data
cashbal = pd.read_csv('./input/raw/POS_CASH_balance.csv')
cashbal.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
cashbal['completed'] = np.where(cashbal.NAME_CONTRACT_STATUS == 'Completed',
    1, 0)
cashbal.head().T

cashbal.NAME_CONTRACT_STATUS.value_counts()

# aggregate to id prev
_aggdict =    { 'MONTHS_BALANCE':'min',
                'CNT_INSTALMENT':'min',
                'CNT_INSTALMENT_FUTURE':'min',
                'SK_DPD':'sum',
                'completed':'sum'
               }
_grouped = cashbal.groupby(['SK_ID_PREV']) # sort=True
cash_agg = _grouped.agg(_aggdict)
cash_agg.head().T



#%% clean up
del inst_agg0
del inst
del revbal
del cashbal
gc.collect()



#%% combine detailed table
aggs = inst_agg.append([rev_agg, cash_agg], sort=False)
aggs = inst_agg.join(rev_agg, how='outer', rsuffix='rev')
aggs = aggs.join(cash_agg, how='outer', rsuffix='cash')
aggs.head(20).T

del inst_agg
del rev_agg
del cash_agg
gc.collect()


#%% read prev and filter to cash & credit
prev = pd.read_csv('./input/raw/previous_application.csv', 
    index_col='SK_ID_PREV')

prev = prev.loc[prev.NAME_CONTRACT_TYPE != "Consumer loans", :]
prev = prev.loc[prev.NAME_CONTRACT_TYPE != "XNA", :]
prev.head().T

prev.NAME_CONTRACT_STATUS.unique()

# merge
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')




#%%
#run maxreduce and save it to parquet




# clean the register and gc()



# In[22]:
train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
train.head().T
