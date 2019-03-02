#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import os
import re

from bokeh.io import output_file, output_notebook
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource
from bokeh.plotting import show
from bokeh.plotting import figure
import numpy as np
import pandas as pd

from helper import display_df_with_bokeh


# In[2]:


get_ipython().system('head -n 10 datasets/cit-HepTh-no-cycles.csv')


# ## Initialise variables

# In[3]:


DATASETS_FOLDER = "datasets"
ABSTRACTS_FOLDER = "abstracts"

ABSTRACTS_FOLDER_PATH = f"{DATASETS_FOLDER}/{ABSTRACTS_FOLDER}/"


# ## Process paper citations

# ### Edges dataset
# 
# 
# **NOTE:** Here we use the clean dataset that was preprocessed in the *preprocessing-dataset.ipynb* notebook

# In[4]:


with open(f"{DATASETS_FOLDER}/cit-HepTh-no-cycles.csv", 'r') as f:
    df = pd.read_csv(f)

# Dropping duplicates
df.drop_duplicates(inplace = True)
    
# Rename columns
df.columns = ['FromNodeId', 'ToNodeId']

df.FromNodeId = df.FromNodeId.map(str)
df.ToNodeId = df.ToNodeId.map(str)


tlds_csv = pd.read_csv(f"{DATASETS_FOLDER}/tlds.csv", header=None, index_col=0, squeeze=True).to_dict()
tlds_info = tlds_csv[1]


# In[5]:


df.head()


# In[6]:


# Paper that cites most papers
out_degree = df.groupby('FromNodeId').count().sort_values('ToNodeId', ascending = False)

display_df_with_bokeh(out_degree, columns={
    "FromNodeId": "Paper",
    "ToNodeId": "Papers cited"
})


# In[7]:


hist, edges = np.histogram(out_degree['ToNodeId'], bins=100, range = [0, 600])

# Create the blank plot
p = figure(plot_height = 500, plot_width = 900, 
           title = 'Citations histogram (out_degree)',
          x_axis_label = 'Papers cited', 
           y_axis_label = 'Papers')

# Add a quad glyph
p.quad(bottom=0, top=hist, 
       left=edges[:-1], right=edges[1:],
       fill_color= 'navy', line_color='white')

# Show the plot
show(p)


# In[8]:


# Paper cited the most -> Most influential
in_degree = df.groupby('ToNodeId').count().sort_values('FromNodeId', ascending = False)

display_df_with_bokeh(in_degree, columns={
    "ToNodeId": "Paper",
    "FromNodeId": "Paper citations"
})


# In[9]:


hist, edges = np.histogram(in_degree['FromNodeId'], bins=100)

# Create the blank plot
p = figure(plot_height = 500, plot_width = 900, 
           title = 'Citations histogram (in_degree)',
          x_axis_label = 'Paper citations', 
           y_axis_label = 'Papers')

# Add a quad glyph
p.quad(bottom=0, top=hist, 
       left=edges[:-1], right=edges[1:],
       fill_color= 'navy', line_color='white')

# Show the plot
show(p)


# In[10]:


degrees = pd.concat([in_degree, out_degree], axis=1, sort=False)
degrees.columns = ['out_degree', 'in_degree']
degrees.index.name = 'paper'

display_df_with_bokeh(degrees, include_index=True)


# In[ ]:




