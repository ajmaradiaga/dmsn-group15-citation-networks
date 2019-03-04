#!/usr/bin/env python
# coding: utf-8

# This notebook reduces the size of our network for visualisation purposes.

# In[1]:


import os
import re

import numpy as np
import pandas as pd

from helper import display_df_with_bokeh


# In[2]:


DATASETS_FOLDER = "datasets"


# In[3]:


# Opening original network with modularity
with open(f"gephi/nodes_with_modularity_degree_centralities.csv", 'r') as f:
    gephi_df = pd.read_csv(f, header=0)


# In[4]:


display_df_with_bokeh(gephi_df, include_index=True)


# In[18]:


gephi_df.sort_values(['indegree'], ascending=False)[:10]


# In[19]:


gephi_df.sort_values(['pageranks'], ascending=False)[:10]


# In[17]:


gephi_df.sort_values(['closnesscentrality'], ascending=False)[:20]


# In[14]:


gephi_df.sort_values(['betweenesscentrality'], ascending=False)[:20]


# In[ ]:




