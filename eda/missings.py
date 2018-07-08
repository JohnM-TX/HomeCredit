
# coding: utf-8

# In[5]:

import pandas as pd
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('./input/raw/application_train.csv')

#eeee

# In[6]:

msno.matrix(train.sample(100))
msno.matrix(train.iloc[0:100, 42:92])
msno.dendrogram(train)
msno.heatmap(train)

