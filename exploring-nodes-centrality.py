#!/usr/bin/env python
# coding: utf-8

# This notebook reduces the size of our network for visualisation purposes.

# In[1]:


import os
import re

import numpy as np
import pandas as pd

from helper import display_df_with_bokeh

pd.set_option('display.width', 1000)


# In[2]:


DATASETS_FOLDER = "datasets"


# In[3]:


# Opening original network with modularity
with open(f"gephi/dynamic/nodes_with_all_network_stats_and_timestamps.csv", 'r') as f:
    gephi_df = pd.read_csv(f, header=0)


# In[4]:


gephi_df.head(10)


# In[8]:


gephi_df[['Id', 'indegree']].sort_values(['indegree'], ascending=False)[:10]


# In[7]:


gephi_df[['Id', 'pageranks']].sort_values(['pageranks'], ascending=False)[:10]


# In[14]:


gephi_df[['Id', 'eigencentrality']].sort_values(['eigencentrality'], ascending=False)[:10]


# In[11]:


gephi_df[['Id', 'closnesscentrality']].sort_values(['closnesscentrality'], ascending=False)[:10]


# In[13]:


gephi_df[['Id', 'betweenesscentrality']].sort_values(['betweenesscentrality'], ascending=False)[:10]


# In[ ]:




