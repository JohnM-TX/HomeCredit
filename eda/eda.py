

#%% get libraries
import os
import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100

import holoviews as hv
hv.extension('bokeh')

from pandas_profiling import ProfileReport
import missingno as msno
from eda.discover import discover


#%% get data
train = pd.read_csv('./input/raw/application_train.csv', index_col='SK_ID_CURR')
test = pd.read_csv('./input/raw/application_test.csv', index_col='SK_ID_CURR')
test['TARGET'] = 2
print (train.shape, '\n', test.shape)

traintest = pd.concat([train, test], sort=False).sort_index()
traintest.head().T


# In[ ]:

ProfileReport(train)


# In[ ]:

traintest.drop('TARGET', axis=1).duplicated(keep=False).sum()


# In[ ]:

msno.matrix(traintest, filter=None, n=0, p=0, sort=None,
           figsize=(25, 15), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
           fontsize=8, labels=None, sparkline=True, inline=True,
           freq=None)


# In[ ]:


get_ipython().run_cell_magic('opts', 'Distribution  [height=240 width=240]', "\nallnums = traintest.drop('TARGET', axis = 1).select_dtypes(include='float')\nplot_list = [hv.Distribution(train[c])*hv.Distribution(test[c]) for c in allnums.columns]\ntrains = hv.Layout(plot_list)\ntrains")


# In[ ]:

all.dtypes


# In[3]:

# cat columns differences
allobjs = traintest.select_dtypes(include='object')
for c in allobjs.columns:
    s1 = set(train[c].unique())
    s2 = set(test[c].unique())
    diff = s1.symmetric_difference(s2)
    for d in diff:
        tnrows = train[train[c] == d].shape[0]
        tsrows = test[test[c] == d].shape[0]
        print('{}, "{}": {} in train, {} in test'.format(c,d,tnrows, tsrows))


# In[9]:

get_ipython().run_cell_magic('opts', 'Scatter  [height=240 width=240, xaxis=None, yaxis=None]', "\nallnums = traintest.drop('TARGET', axis = 1).select_dtypes(include='float')\nplot_list = [hv.Scatter(traintest, 'SK_ID_CURR', c) for c in allnums.columns[1:2]] ####\ntrains = hv.Layout(plot_list)\ntrains")


# In[13]:

get_ipython().run_cell_magic('opts', 'Scatter  [height=240 width=240, xaxis=None, yaxis=None]', "\nallcats = traintest.drop('TARGET', axis = 1).select_dtypes(include='object')\nfor c in allcats.columns:\n    traintest[c] = traintest[c].astype('category').cat.codes\n")


# In[16]:


plot_list = [hv.Scatter(traintest, 'TARGET', c) for c in allcats.columns[1:3]] ####
trains = hv.Layout(plot_list)
trains


# In[25]:


allcats.columns[0:2]


# In[77]:


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


# In[87]:



get_ipython().run_cell_magic('opts', "Bars.Grouped [group_index='set']", "%%opts Bars [group_index=1 xrotation=45 width=480 show_legend=False tools=['hover']]\n%%opts Bars (color=Cycle('Set2'))\n\nallcats = traintest.drop('TARGET', axis = 1).select_dtypes(include='object')\nplot_list = []\nfor c in allcats.columns:\n    plot = catcompare(c)\n    plot_list.append(plot)\n\nhv.Layout(plot_list)")


# In[85]:


import seaborn as sns
sns.set_style()
tips = sns.load_dataset("tips")
ax = sns.countplot(x="CODE_GENDER", data=train)


# In[83]:


tips

