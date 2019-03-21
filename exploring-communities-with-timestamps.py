#!/usr/bin/env python
# coding: utf-8

# This notebook reduces the size of our network for visualisation purposes.

# In[1]:


import os
import re

import numpy as np
import pandas as pd

from helper import display_df_with_bokeh

pd.set_option('display.width', 400)


# In[2]:


DATASETS_FOLDER = "datasets"


# In[3]:


# Opening original network with modularity
with open(f"gephi/dynamic/nodes_with_all_network_stats_and_timestamps.csv", 'r') as f:
    gephi_df = pd.read_csv(f, header=0)


# In[8]:


# Calculate members per group
gephi_df.groupby('componentnumber').count().sort_values('Id', ascending = False).head(10)


# In[5]:


# Calculate members per group
gephi_df.groupby('modularity_class').count().sort_values('Id', ascending = False).head(10)


# In[21]:


# Calculate citations per group
gephi_df.groupby('modularity_class').sum().sort_values('indegree', ascending = False).head(10)


# In[25]:


gephi_df[(gephi_df.Id == 9711200) | (gephi_df.Id == 9802150) | (gephi_df.Id == 9802109) | (gephi_df.Id == 9407087)| (gephi_df.Id == 9610043)]


# In[26]:


gephi_df.sort_values('date')


# In[10]:


# Calculate records per group
modularity_df = gephi_df.groupby('modularity_class').count().sort_values('Id', ascending = False)

# Only taking in consideration where groups have more than 10 nodes
modularity_mt_nodes_df = modularity_df[modularity_df.Id >= 100]

# display_df_with_bokeh(modularity_mt_nodes_df, include_index=True)
print(modularity_mt_nodes_df.shape)

modularity_mt_nodes_df


# In[11]:


print("Total nodes that form part of communities with more than 100 members: ", modularity_mt_nodes_df.sum()['Id'])


# In[12]:


print("Total nodes that form part of communities with less than 100 members: ", modularity_df[modularity_df.Id < 100].sum()['Id'])


# In[13]:


modularity_df[modularity_df.Id < 100]


# In[ ]:


gephi_df[gephi_df['modularity_class'] == 2]


# In[ ]:


total_nodes = modularity_mt_nodes_df.sum()['Id']
total_nodes


# In[ ]:


# DF will contain the most important nodes of each modularity class
nodes_per_modularity_class_df = None


# In[ ]:


for row in modularity_mt_nodes_df.iterrows():
    
    modularity = row[0]
    
    total_nodes_in_class = modularity_mt_nodes_df.loc[modularity]['Id']
    proportion_of_network = (total_nodes_in_class / total_nodes)
    subset_of_nodes = int(round(proportion_of_network * total_nodes_in_class)) + 1
    
    print(f"Modularity class: {modularity} - Total nodes in class: {total_nodes_in_class} - Proportion of final network: {proportion_of_network} - New total {subset_of_nodes}")
    
    aux = gephi_df[gephi_df.modularity_class == modularity].sort_values('Degree', ascending=False)[:subset_of_nodes]
    
    if nodes_per_modularity_class_df is None:
        nodes_per_modularity_class_df = aux
    else:
        nodes_per_modularity_class_df = nodes_per_modularity_class_df.append(aux, ignore_index=True)


# In[ ]:


display_df_with_bokeh(nodes_per_modularity_class_df)
print("Most important nodes of our network: ", nodes_per_modularity_class_df.shape[0])


# In[ ]:


with open(f"{DATASETS_FOLDER}/cit-HepTh.txt", 'r') as f:
    hepth_df = pd.read_csv(f,sep='\t',skiprows=(0,1,2))
    
# Rename columns
hepth_df.columns = ['FromNodeId', 'ToNodeId']


# In[ ]:


display_df_with_bokeh(hepth_df)


# In[ ]:


nodes_with_modularity_df = nodes_per_modularity_class_df[['Id', 'modularity_class']]
display_df_with_bokeh(nodes_with_modularity_df)


# In[ ]:


# Join the original network with our most important nodes with modularity
hepth_df_with_modularity = hepth_df.merge(nodes_with_modularity_df, how="outer", left_on = 'FromNodeId', right_on = 'Id')

hepth_df_with_modularity.columns = ['FromNodeId', 'ToNodeId', 'Id', 'modularity_from']
hepth_df_with_modularity.drop(['Id'], axis=1, inplace=True)

hepth_df_with_modularity = hepth_df_with_modularity.merge(nodes_with_modularity_df, how="outer", left_on = 'ToNodeId', right_on = 'Id')

hepth_df_with_modularity.columns = ['FromNodeId', 'ToNodeId', 'modularity_from', 'Id', 'modularity_to']
hepth_df_with_modularity.drop(['Id'], axis=1, inplace=True)


# In[ ]:


# Remove all edges that have a node that was not found in nodes_with_modularity_df
hepth_df_with_modularity.dropna(inplace=True)
hepth_df_with_modularity = hepth_df_with_modularity.astype(int)


# In[ ]:


display_df_with_bokeh(hepth_df_with_modularity)
print(hepth_df_with_modularity.shape)


# In[ ]:


# Save our new reduced network with most important nodes
hepth_df_with_modularity.to_csv("gephi/reduced_size_network.csv")


# In[ ]:




