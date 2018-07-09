

#%% get libraries
import os
import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100

import holoviews as hv
hv.extension('bokeh')

import seaborn as sns

from pandas_profiling import ProfileReport
import missingno as msno
from eda.discover import discover




#%% get data and peek
train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
test = pd.read_csv('./input/raw/application_test.csv', index_col='SK_ID_CURR')
test['TARGET'] = 2
print (train.shape, '\n', test.shape)
train.head()
test.head()

traintest = pd.concat([train, test], sort=False).sort_index()
traintest.head().T

#%% for demo only: reduce features...
dropstrings = ['FLAG', 'AVG', 'MODE', 'MEDI']

trainlite = train.copy()
for d in dropstrings:
    trainlite.drop(trainlite.filter(like = d).columns, axis = 1, inplace=True)
trainlite.shape

traintestlite = traintest[trainlite.columns]
traintestlite.shape



################################
#### Perform some EDA ####
################################

#%% get automated description
ProfileReport(trainlite)


#%% check for patterns
traintestlite.AMT_CREDIT.sample(frac = 0.05).plot()
traintestlite.reset_index().SK_ID_CURR.plot()
trainlite.TARGET.sample(frac = 0.05).plot()

#%% check duplicate rows and constant columns
traintestlite.drop('TARGET', axis=1).duplicated(keep=False).sum()
consts = traintestlite.nunique(axis=0)
consts[consts ==1]

#%% check correlated columns and scatters
corr = traintestlite.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
corr
pd.scatter_matrix(traintestlite.iloc[:, 2:6].sample(frac = 0.01))

#%% check missing variable structure
msno.matrix(traintest, filter=None, n=0, p=0, sort=None,
           figsize=(25, 15), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
           fontsize=8, labels=None, sparkline=True, inline=True,
           freq=None)


#%% check distros for numericals
get_ipython().run_cell_magic('opts', 'Distribution  [height=240 width=240]
allnums = traintestlite.drop('TARGET', axis = 1).select_dtypes(include='float')
plot_list = [hv.Distribution(train[c])*hv.Distribution(test[c]) for c in allnums.columns]
pltmat = hv.Layout(plot_list)
pltmat

# see dotplot from kernels for more numericals

#%% check cat columns differences
allobjs = traintest.select_dtypes(include='object')
for c in allobjs.columns:
    s1 = set(train[c].unique())
    s2 = set(test[c].unique())
    diff = s1.symmetric_difference(s2)
    for d in diff:
        tnrows = train[train[c] == d].shape[0]
        tsrows = test[test[c] == d].shape[0]
        print('{}, "{}": {} in train, {} in test'.format(c,d,tnrows, tsrows))


#%% run scatterplots v target:

get_ipython().run_cell_magic('opts', 'Scatter  [height=240 width=240, xaxis=None, yaxis=None]
allnums = traintest.drop('TARGET', axis = 1).select_dtypes(include='float')
plot_list = [hv.Scatter(traintest, 'SK_ID_CURR', c) for c in allnums.columns[1:2]] ####
trains = hv.Layout(plot_list)
trains

get_ipython().run_cell_magic('opts', 'Scatter  [height=240 width=240, xaxis=None, yaxis=None]
allcats = traintest.drop('TARGET', axis = 1).select_dtypes(include='object')
for c in allcats.columns:
    traintest[c] = traintest[c].astype('category').cat.codes
    plot_list = [hv.Scatter(traintest, 'TARGET', c)] ####
trains = hv.Layout(plot_list)
trains


#%%Look at feature ratios by target value:
def catcompare(feature):
    h = train[feature].value_counts(normalize=True)
    h_df = pd.DataFrame({'cat':h.index, 'pct':h.values})
    h_df['set'] = 'train'
    i = test[feature].value_counts(normalize=True)
    i_df = pd.DataFrame({'cat':i.index, 'pct':i.values})
    i_df['set'] = 'test'
    j_df = pd.concat([h_df,i_df])

    key_dimensions   = [('cat', 'Ct'), ('set', 'St')]
    value_dimensions = [('pct', 'Pct')]
    macro = hv.Table(j_df)

    plot = macro.to.bars(kdims=['set', 'cat'], vdims='pct', groupby=[], label = feature)
    return plot

get_ipython().run_cell_magic('opts', "Bars.Grouped [group_index='set']
opts Bars [group_index=1 xrotation=45 width=480 show_legend=False tools=['hover']] 
%%opts Bars (color=Cycle('Set2')) 
allcats = traintest.drop('TARGET', axis = 1).select_dtypes(include='object') 
plot_list = [] 
for c in allcats.columns: 
plot = catcompare(c) 
plot_list.append(plot)
hv.Layout(plot_list)")


#%% Much simpler countplot
import seaborn as sns
sns.set_style()
tips = sns.load_dataset("tips")
ax = sns.countplot(x="CODE_GENDER", data=train)