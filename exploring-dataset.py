#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re

from bokeh.io import output_file, output_notebook
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource
from bokeh.plotting import show
from bokeh.plotting import figure
from fastparquet import write
import numpy as np
import pandas as pd
import snappy

from helper import display_df_with_bokeh


# In[2]:


get_ipython().system('head -n 10 datasets/cit-HepTh.txt')


# ## Initialise variables

# In[3]:


DATASETS_FOLDER = "datasets"
ABSTRACTS_FOLDER = "abstracts"

ABSTRACTS_FOLDER_PATH = f"{DATASETS_FOLDER}/{ABSTRACTS_FOLDER}/"


# ## Processing functions

# In[4]:


def extract_text_from_abstract(text):
    info = {}
    fields = ['Date:', "From:", "Title:", "Authors:", "Comments:", "Subj-class:", "Journal-ref:"]
    
    for field in fields:
        match = re.search(f"[\n\r].*{field}\s*([^\n\r]*)", text, re.I)
        value = None
        if match is not None:
            value = match.group(1)
        
        info[field.replace(':', '').lower()] = value
        
    return info

ignore_emails = ['g@c']
ignore_tlds = ['g@c', '']
ignore_domains = ['c', '']

def domain_and_tld_from_email(email):
    domain = None
    tld = None
    email = email.lower()
    if '.' in email:
                        
        # Remove cases when email finishes with a period
        if email[-1] == '.':
            email = email[:-1]

        domain = email.split('@')[1].lower().strip()

        if '.ac.' in domain or '.co.' in domain or '.edu.' in domain or '.gov.' in domain or '.com.' in domain:
            domain = ".".join(domain.split('.')[-3:])
        else:
            domain = ".".join(domain.split('.')[-2:])

        tld  = email.split('.')[-1].lower().strip()

        if tld in ignore_tlds:
            tld = None

        if domain in ignore_domains:
            domain = None
        
    
    return domain, tld

# Code below from https://stackoverflow.com/a/40449726/9527459
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]


# ## Process paper citations

# In[5]:


with open(f"{DATASETS_FOLDER}/cit-HepTh.txt", 'r') as f:
    df = pd.read_csv(f,sep='\t',skiprows=(0,1,2))
    
# Rename columns
df.columns = ['FromNodeId', 'ToNodeId']

tlds_csv = pd.read_csv(f"{DATASETS_FOLDER}/tlds.csv", header=None, index_col=0, squeeze=True).to_dict()
tlds_info = tlds_csv[1]


# In[6]:


display_df_with_bokeh(df[150:250])


# In[7]:


# Paper that cites most papers
out_degree = df.groupby('FromNodeId').count().sort_values('ToNodeId', ascending = False)

display_df_with_bokeh(out_degree, columns={
    "FromNodeId": "Paper",
    "ToNodeId": "Papers cited"
})


# In[8]:


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


# In[9]:


# Paper cited the most -> Most influential
in_degree = df.groupby('ToNodeId').count().sort_values('FromNodeId', ascending = False)

display_df_with_bokeh(in_degree, columns={
    "ToNodeId": "Paper",
    "FromNodeId": "Paper citations"
})


# In[10]:


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


# In[11]:


degrees = pd.concat([in_degree, out_degree], axis=1, sort=False)
degrees.columns = ['out_degree', 'in_degree']
degrees.index.name = 'paper'

display_df_with_bokeh(degrees, include_index=True)


# # Process paper abstracts

# In[12]:


abstracts_info = {}

for dir_name in os.listdir(f"{ABSTRACTS_FOLDER_PATH}"):
    try:
        year = int(dir_name)
        
        for f_name in os.listdir(f"{ABSTRACTS_FOLDER_PATH}/{year}"):
            with open(f"{ABSTRACTS_FOLDER_PATH}/{year}/{f_name}", 'r') as f:
                abstract = f.read()
                
                # Parts of the abstract
                abstract_parts = abstract.split('\\\\')
                
                paper_description = (abstract_parts[2] if len(abstract_parts) > 1 else "").strip()
                
                # Process emails
                emails_found = re.findall(r'[\w\.-]+@[\w\.-]+', abstract)
                
                emails = []
                
                for email in emails_found:
                    email = email.lower()
                    
                    if '.' in email:
                        # Remove cases when email finishes with a period
                        if email[-1] == '.':
                            email = email[:-1]
                        
                        emails.append(email)
                
                
                key = int(f_name.replace(".abs", ""))
                
                abstracts_info[key] = {
                    "emails": list(set(emails)),
#                     "tlds": list(set(tlds)),
#                     "domains": list(set(domains)),
                    "description": paper_description
                }
                
                abstracts_info[key].update(extract_text_from_abstract(abstract))
                
                
    except ValueError:
        pass 


# In[13]:


papers = pd.DataFrame.from_dict(abstracts_info, orient='index')

papers = pd.concat([papers, degrees], axis=1, sort=False)


# In[14]:


papers.head()


# In[15]:


more_than_one_email =  [True if len(e) == 0 else False for e in papers.emails]
papers[more_than_one_email]


# # Enrich Paper citations

# In[16]:


citations = df.join(papers[['emails']], on='FromNodeId')
citations.columns = ['FromNodeId', 'ToNodeId', 'emails_from']

citations = citations.join(papers[['emails']], on='ToNodeId')
citations.columns = ['FromNodeId', 'ToNodeId', 'emails_from', 'emails_to']

explode_columns = ['emails_from', 'emails_to']

for ec in explode_columns:
    citations = explode(citations, [ec])

citations.drop_duplicates(inplace=True)
    
display_df_with_bokeh(citations.head(20))


# In[17]:


citations["domain_from"], citations["tld_from"] = zip(*citations["emails_from"].map(domain_and_tld_from_email))
citations["domain_to"], citations["tld_to"] = zip(*citations["emails_to"].map(domain_and_tld_from_email))

display_df_with_bokeh(citations)


# ## TLD Aggregation

# In[18]:


# pd.Series([item for sublist in papers.tlds for item in sublist])

# Flatten tlds
tld_series = pd.concat([citations.tld_from, citations.tld_to], axis=0) 

# Count different values
tld_df = tld_series.value_counts().sort_index().rename_axis('tld').reset_index(name='count')

# Add description column
tld_df['tlds_description'] = tld_df['tld'].map(lambda x: tlds_info[x] if x in tlds_info else None)


# In[19]:


display_df_with_bokeh(tld_df.sort_values('count', ascending=False))


# ## Domain Aggregation _a.k.a. most influential institutions / labs_

# In[20]:


#pd.Series([item for sublist in papers.domains for item in sublist])

# Flatten tlds
domain_series = pd.concat([citations.domain_from, citations.domain_to], axis=0) 

# Count different values
domain_df = domain_series.value_counts().sort_index().rename_axis('domain').reset_index(name='count')

# Sorting by count
display_df_with_bokeh(domain_df.sort_values('count', ascending=False))


# ## Institutions per country
# ![alt text](visualisation/screenshots/institutions_per_country.jpeg "Title")

# ## Save Dataframes

# In[ ]:


# Save dataframes
write(f"{DATASETS_FOLDER}/parquet/paper_citations.pq", citations)
write(f"{DATASETS_FOLDER}/parquet/papers.pq", papers)
write(f"{DATASETS_FOLDER}/parquet/tld_aggregation.pq", tld_df)
write(f"{DATASETS_FOLDER}/parquet/domain_aggregation.pq", domain_df)


# In[ ]:




