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
from fastparquet import write
import numpy as np
import pandas as pd
import snappy

from helper import display_df_with_bokeh


# In[2]:


get_ipython().system('head -n 10 datasets/cit-HepTh.txt')


# ## Initialise variables

# In[5]:


DATASETS_FOLDER = "datasets"
ABSTRACTS_FOLDER = "abstracts"

ABSTRACTS_FOLDER_PATH = f"{DATASETS_FOLDER}/{ABSTRACTS_FOLDER}/"


# ## Processing functions

# In[12]:


TIME_REPLACEMENT_REGEX = [
    ' [\d]+:\d\d[:\d\d[.\d\d]*]*',
    '\(\d*kb\)',
    '[\d]+:\d\d:\d\d',
    '[+-][\d]+:\d\d',
    ' GMT[+-]\d\d\d\d ',
    '\s[+-]\d+\s',
    '[+-]*0\d\d\d ',
    '[+-][\d]+:\d\d',
    '\s+',
]

DATE_FORMATS = [
    '%a %d %b %Y',
    '%a %d %b %y',
    '%a %d %b %y',
    '%d %b %y',
    '%d %b %Y',
    '%a %d %B %Y',
    '%a %d %B %y',
    '%a %B %d %Y',
    '%a %b %d %Y',
    '%a %B %d %y',
    '%d %B %y',
    '%d %B %Y',
    '%d/%m/%y',
    '%d-%b-%Y'
]

def date_from_str(s, print_log=False):
    """Process the date that it is in the paper abstract"""
    replaced = s
    log = []
    # print(s, '\n===============')
    for rex in TIME_REPLACEMENT_REGEX:
        replaced = re.sub(rex, ' ', replaced)
        log.append(rex + ' -> ' + replaced)
    
    aux = re.sub(' ["A-Za-z ()]+$', '', replaced.strip())
    aux = re.sub('MET|EDT|CDT|NV|GMT', '', aux.strip())
    
    aux = aux.replace('December', 'Dec').replace('September', 'Sep')
    
    aux = re.sub('\s+', ' ', aux.strip())
    
    aux = aux.replace(',','').replace('"', '').replace('', '').replace('(', '')[:15].title().strip()
    
    #print(aux)
    
    val = None
    
    for fmt in DATE_FORMATS:
        try:
            val = datetime.strptime(aux, fmt)
            break
        except ValueError:
            pass
    
    if val is None:
        print(s, aux)
        if print_log:
            for l in log:
                print(l)
    
    return val


def extract_text_from_abstract(text):
    """Process the contents of paper abstracts"""
    info = {}
    fields = ['Date:', "From:", "Title:", "Authors:", "Comments:", "Subj-class:", "Journal-ref:"]
    
    for field in fields:
        match = re.search(f"[\n\r].*{field}\s*([^\n\r]*)", text, re.I)
        value = None
        if match is not None:
            value = match.group(1)
            
            if field == 'Date:':
                aux = value
                
                val = date_from_str(aux)
                
                value = val
                
                if val is None:
                    print(aux, value)
                
        
        info[field.replace(':', '').lower()] = value
        
    return info


ignore_emails = ['g@c']
ignore_tlds = ['g@c', '']
ignore_domains = ['c', '']

def domain_and_tld_from_email(email):
    """Process domain and tld from email addresses"""
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

# ### Edges dataset

# In[7]:


with open(f"{DATASETS_FOLDER}/cit-HepTh.txt", 'r') as f:
    df = pd.read_csv(f,sep='\t',skiprows=(0,1,2))

# Dropping duplicates
df.drop_duplicates(inplace = True)
    
# Rename columns
df.columns = ['FromNodeId', 'ToNodeId']

df.FromNodeId = df.FromNodeId.map(str)
df.ToNodeId = df.ToNodeId.map(str)

df.drop_duplicates(inplace=True)

tlds_csv = pd.read_csv(f"{DATASETS_FOLDER}/tlds.csv", header=None, index_col=0, squeeze=True).to_dict()
tlds_info = tlds_csv[1]


# ### Paper - Dates dataset

# In[8]:


with open(f"{DATASETS_FOLDER}/cit-HepTh-dates.txt", 'r') as f:
    df_dates = pd.read_csv(f,sep='\t',names=['papers', 'date'],skiprows=(1))

df_dates.date = df_dates.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_dates.papers = df_dates.papers.map(str)

df_dates.papers = df_dates.papers.apply(lambda x: x[2:] if x.startswith("111") and len(x) > 5 else x)

# Papers can have multiple dates. We select the minimum date
df_dates = df_dates.groupby(['papers']).min()
df_dates = df_dates.reset_index()

# Remove duplicates
df_dates.drop_duplicates(inplace=True)


# In[9]:


df_dates[(df_dates.papers == '11111056') | (df_dates.papers == '111056')]

xy = df_dates.groupby(['papers']).count()
xy[xy.date > 3]

df_dates[(df_dates.papers == '111001') | (df_dates.papers == '1001') | (df_dates.papers == '9311042') | (df_dates.papers == '119311042')]


# Now, we join our edges dataset with the dates. We will notice that not all papers have dates in the cit-HepTh-dates.txt

# In[10]:


paper_dates = pd.merge(df, df_dates, how = 'left', left_on = 'FromNodeId', right_on = 'papers') # df.join(df_dates[['date']], on='FromNodeId')

paper_dates = paper_dates[['FromNodeId', 'ToNodeId', 'date']]
paper_dates.columns = ['FromNodeId', 'ToNodeId', 'date_from']

paper_dates = pd.merge(paper_dates, df_dates, how = 'left', left_on = 'ToNodeId', right_on = 'papers') # df.join(df_dates[['date']], on='FromNodeId')

paper_dates = paper_dates[['FromNodeId', 'ToNodeId', 'date_from', 'date']]
paper_dates.columns = ['FromNodeId', 'ToNodeId', 'date_from', 'date_to']

paper_dates.head(10)


# As there are too many papers without a date, we will use the dates in the paper abstracts to identify if an old paper references a future paper3

# ### Paper - abstracts

# In[13]:


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
                    "description": paper_description
                }
                
                abstracts_info[key].update(extract_text_from_abstract(abstract))
                
    except ValueError:
        pass 


# In[ ]:



# s = [
#     "Sun Oct 17 19:16:57 1993", 'Fri, 28 Apr 1995 18:52:55   (7kb)',
#      'Fri, 28 Apr 1995 20:05:27 +0200 (METDST)   (18kb)',
#      'Tue, 17 Oct 1995 21:13:11 -0500 (CDT)   (10kb)',
# 'Mon, 13 Nov 95 16:06 GMT-0600   (11kb)',
# 'Thu, 9 Nov 1995 22:36:46 +0300   (10kb)',
# 'Thu, 8 Jun 1995 20:11:22 +0300 (WET)   (26kb)',
#      'Fri, 19 May 95 17:41:19 BST   (8kb)',
# 'Tue, 21 Mar 1995 13:51:34 +0100   (118kb)',
# 'Wed, 23 Aug 1995 11:23:20 +0900   (6kb)', 
# 'Tue, 17 Oct 1995 15:09:00 +0100   (28kb)',
#      'Sun, 27 Aug 1995 17:48:55 +0900 (JST)   (33kb)',
#      'Tue, 21 Feb 1995 9:47:28 -0600 (CST)   (10kb)',
# 'Thu, 21 Sep 1995 14:00:07 +0200 (MET DST)   (9kb)',
# 'Thu, 30 Nov 95 15:01:48 -0500   (17kb)',
#      'Wed, 13 Sep 0 21:31:44 "KST   (9kb)',
#      'Sat, 10 Jun 95 22:02:15 GMT+9:00   (9kb)',
#      'Wed, 30 Aug 95 17:25:31+0900   (12kb)',
#      'Sun, 24 Dec 95 21:39:01 -2359   (15kb)',
# 'Mon, 01 May 95 17:45:12 +1000   (6kb)',
#      'Sun, 31 Jan 93 14:33:42+050   (9kb)',
# 'Fri, 9 Apr 93 17:51:18-010   (14kb)',
#      'Thu, 11 Oct 2001 08:13:35 GMT   (8kb)',
# 'Thu, 11 Oct 2001 08:13:35 GMT',
#      'Tue, 26 Mar 96 00:20:35 GMT-0600   (144kb)',
# 'Wed, 24 Jan 96 12:24 0200   (11kb)',
# 'Wed, 24 Jan 96 12:24 0200   (11kb)',
#     'Wed, 17 Feb 93 13:36:51 NV   (11kb)',
# 'Thu, 30 May 96 18:27:53 +12000   (21kb)',
#      '1 September 1992 15:32:14 CDT   (17kb)',
# '14 December 1992 07:58:35 CST   (30kb)',
# 'Thu, 20 Jan 94 10:46:25+050 (12kb)',
#      'Wed, 24 Jan 96 12:24 0200   (11kb)',
#      '01 Oct 1992 13:05:29 +0000 (N)   (12kb)',
# 'Tue Dec 31 23:54:17 MET 1991 +0100   (37kb)',
#      'Fri, 1 Sep 1995 12:10:13 +0200 (MET DST)   (47kb)',
#      'Wed, 13 Sep 0 21:31:44 "KST   (9kb)',
# '28-JAN-1994 14:29:23.91 TST   (8kb)',
# '9-MAR-1993 14:33:25.98   (110kb)',
# 'Wed, 17 Feb 93 13:36:51 NV   (11kb)',
#     'Tue Dec 31 23:54:17 MET 1991 +0100   (37kb)',
# '1 September 1992 15:32:14 CDT   (17kb)',
# '14-JUL-1992 15:04:09.32 TST   (7kb)',
#      '14 December 1992 07:58:35 CST   (30kb)'
#     ]

# for y in s:
#     v = date_from_str(y)
#     print(y, ' -> ', v)


# In[14]:


abstracts = pd.DataFrame.from_dict(abstracts_info, orient='index')

abstracts.index = abstracts.index.map(str)

abstract_dates = abstracts.date.reset_index()
abstract_dates.columns = ['papers', 'date']
abstract_dates.papers = abstract_dates.papers.map(str)

abstract_dates.head()


# In[105]:


abstracts.head(10)


# In[106]:


abstracts[['title', 'description', 'date']].head(100)


# In[15]:


# Abstracts that contain more than one email
more_than_one_email =  [True if len(e) == 0 else False for e in abstracts.emails]
abstracts[more_than_one_email].shape[0]


# Lets enrich the nodes with dates

# In[16]:


paper_dates = pd.merge(df, abstract_dates, how = 'left', left_on = 'FromNodeId', right_on = 'papers') # df.join(df_dates[['date']], on='FromNodeId')
# del paper_dates.papers

paper_dates = paper_dates[['FromNodeId', 'ToNodeId', 'date']]
paper_dates.columns = ['FromNodeId', 'ToNodeId', 'date_from']

paper_dates = pd.merge(paper_dates, abstract_dates, how = 'left', left_on = 'ToNodeId', right_on = 'papers') # df.join(df_dates[['date']], on='FromNodeId')
# del paper_dates.papers

paper_dates = paper_dates[['FromNodeId', 'ToNodeId', 'date_from', 'date']]
paper_dates.columns = ['FromNodeId', 'ToNodeId', 'date_from', 'date_to']

paper_dates.head()


# We've discovered that in our network, there are some edges where the date of the FromNode < the date of the ToNode. Given that this is a citation network, it is not possible to have old papers that cite new papers, e.g. a paper from 1994 can not cite a paper from 1995. Some examples below.

# In[17]:


paper_dates[paper_dates.date_from < paper_dates.date_to].head(10)


# Lets compare the size of our network without cycles against the original

# In[20]:


edges_without_cycles = paper_dates[paper_dates.date_from > paper_dates.date_to][['FromNodeId', 'ToNodeId']]

print('Edges (original): ', df.shape[0])
print('Edges (after removing cycles): ', edges_without_cycles.shape[0])


# Now that we have a clean network, lets update our original dataset and save it

# In[21]:


df = edges_without_cycles


# In[22]:


df.to_csv(f"{DATASETS_FOLDER}/cit-HepTh-no-cycles.csv", index=False)


# In[40]:


edges_with_dates = paper_dates[paper_dates.date_from > paper_dates.date_to]
edges_with_dates.to_csv(f"{DATASETS_FOLDER}/cit-HepTh-with-dates.csv", index=False)


# In[59]:


nodes = edges_with_dates[['FromNodeId', 'date_from']]
nodes.columns = ['Id', 'date']

# print(nodes.shape)

to_nodes = edges_with_dates[['ToNodeId', 'date_to']]
to_nodes.columns = ['Id', 'date']

# Create unique dataset with all nodes
nodes = pd.concat([nodes, to_nodes])
nodes.drop_duplicates(inplace=True)

print("Total unique papers: ", nodes.shape[0])


# Lets find the last citation that a paper has to know for how long that paper has been relevant.

# In[61]:


nodes_citations = pd.merge(nodes, edges_with_dates[['FromNodeId', 'date_from', 'ToNodeId']], how = 'left', left_on = 'Id', right_on = 'ToNodeId')

nodes_with_last_citation = nodes_citations[['Id', 'date', 'date_from']].groupby(['Id', 'date']).max().reset_index()

#nodes_citations.head(100)
nodes_with_last_citation.columns = ['Id', 'date', 'last_citation']

# If the paper was not cited, set the last_citation value to when it was published
nodes_with_last_citation.last_citation.fillna(nodes_with_last_citation.date, inplace=True)

#
nodes_with_last_citation.head(100)


# In[62]:


nodes_with_last_citation.to_csv(f"{DATASETS_FOLDER}/cit-HepTh-nodes.csv", index=False)


# In[118]:


nodes_last_citation_ix = nodes_with_last_citation.set_index('Id', inplace=False)
nodes_last_citation_ix['last_citation']


# In[124]:


abstract_with_last_citation = abstracts.join(nodes_last_citation_ix[['last_citation']])
abstract_with_last_citation.index.name = 'paper'

abstract_with_last_citation.to_csv(f"{DATASETS_FOLDER}/abstracts_with_last_citation.csv")


# In[133]:


abstracts.index.name = 'paper'

edges_with_from_abstracts = pd.merge(edges_with_dates, abstract_with_last_citation, how = 'left', left_on = 'FromNodeId', right_on = 'paper')
edges_with_from_abstracts.to_csv(f"{DATASETS_FOLDER}/edges_with_from_abstracts.csv", index = False)


# ## Network statistics

# In[23]:


# Paper that cites most papers
out_degree = df.groupby('FromNodeId').count().sort_values('ToNodeId', ascending = False)

display_df_with_bokeh(out_degree, columns={
    "FromNodeId": "Paper",
    "ToNodeId": "Papers cited"
})


# In[24]:


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


# In[25]:


# Paper cited the most -> Most influential
in_degree = df.groupby('ToNodeId').count().sort_values('FromNodeId', ascending = False)

display_df_with_bokeh(in_degree, columns={
    "ToNodeId": "Paper",
    "FromNodeId": "Paper citations"
})


# In[26]:


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


# In[27]:


degrees = pd.concat([in_degree, out_degree], axis=1, sort=False)
degrees.columns = ['out_degree', 'in_degree']
degrees.index.name = 'paper'

display_df_with_bokeh(degrees, include_index=True)


# # Enrich Paper citations

# In[28]:


citations = df.join(abstracts[['emails']], on='FromNodeId')
citations.columns = ['FromNodeId', 'ToNodeId', 'emails_from']

citations = citations.join(abstracts[['emails']], on='ToNodeId')
citations.columns = ['FromNodeId', 'ToNodeId', 'emails_from', 'emails_to']

explode_columns = ['emails_from', 'emails_to']

for ec in explode_columns:
    citations = explode(citations, [ec])

citations.drop_duplicates(inplace=True)
    
display_df_with_bokeh(citations.head(20))


# In[29]:


citations["domain_from"], citations["tld_from"] = zip(*citations["emails_from"].map(domain_and_tld_from_email))
citations["domain_to"], citations["tld_to"] = zip(*citations["emails_to"].map(domain_and_tld_from_email))

display_df_with_bokeh(citations)


# In[30]:


citations.head()


# In[31]:


# Example of papers with multiple email addresses
citations[(citations['FromNodeId'] == 9903234) & (citations['ToNodeId'] == 9708001)]


# ## TLD Aggregation

# In[32]:


# pd.Series([item for sublist in papers.tlds for item in sublist])

# Flatten tlds
tld_series = pd.concat([citations.tld_from], axis=0) 

# Count different values
tld_df = tld_series.value_counts().sort_index().rename_axis('tld').reset_index(name='count')

# Add description column
tld_df['tlds_description'] = tld_df['tld'].map(lambda x: tlds_info[x] if x in tlds_info else None)


# In[33]:


display_df_with_bokeh(tld_df.sort_values('count', ascending=False))


# ## Papers in network by institution

# In[34]:


all_from = citations[['FromNodeId','domain_from']]
all_from.columns = ['Paper', 'domain']

all_to = citations[['ToNodeId','domain_to']]
all_to.columns = ['Paper', 'domain']

# Join all nodes and remove duplicates.
# As a paper can be as a FromNode and ToNode, we will remove duplicates
all_nodes = all_from.append(all_to).drop_duplicates()

print(all_nodes.shape)

# Count by domain
domains_contributions = all_nodes.domain.value_counts().sort_index().rename_axis('domain').reset_index(name='count')

# Sorting by count
display_df_with_bokeh(domains_contributions.sort_values('count', ascending=False))


# ## Most cited institution / lab

# In[35]:


# Only taking in consideration the domain of the ToNode
domain_series = pd.concat([citations.domain_to], axis=0) 

# Count different values
domain_df = domain_series.value_counts().sort_index().rename_axis('domain').reset_index(name='count')

# Sorting by count
display_df_with_bokeh(domain_df.sort_values('count', ascending=False))


# # Institutions / labs that cited the most

# In[36]:


# Flatten tlds
domain_series_to = pd.concat([citations.domain_from], axis=0) 

# Count different values
domain_df_to = domain_series_to.value_counts().sort_index().rename_axis('domain').reset_index(name='count')

# Sorting by count
display_df_with_bokeh(domain_df_to.sort_values('count', ascending=False))


# ## Institutions per country
# ![alt text](visualisation/screenshots/institutions_per_country.jpeg "Title")

# In[66]:


citations.head()


# ## QMUL contributions

# In[89]:


qmul_citations = citations[
    (citations.domain_from == 'qmul.ac.uk') | (citations.domain_to == 'qmul.ac.uk') | 
    (citations.domain_from == 'qmw.ac.uk') | (citations.domain_to == 'qmw.ac.uk')
]

qmul_citations_from = qmul_citations[
    (qmul_citations.domain_from == 'qmul.ac.uk') | 
    (qmul_citations.domain_from == 'qmw.ac.uk')][['FromNodeId']#,'domain_from']
]
qmul_citations_from.columns = ['Paper']

qmul_citations_to = qmul_citations[
    (qmul_citations.domain_to == 'qmul.ac.uk') | 
    (qmul_citations.domain_to == 'qmw.ac.uk')][['ToNodeId']#,'domain_to']
]
qmul_citations_to.columns = ['Paper']

# Join all nodes and remove duplicates.
# As a paper can be as a FromNode and ToNode, we will remove duplicates
qmul_papers = qmul_citations_from.append(qmul_citations_to).drop_duplicates()

print(qmul_papers.shape)


# In[94]:


qmul_papers.Paper = qmul_papers.Paper.map(str)


# In[92]:


in_degree.head()


# In[100]:


qmul_papers_in_degree = pd.merge(in_degree, qmul_papers, how = 'left', left_on = 'ToNodeId', right_on = 'Paper') # df.join(df_dates[['date']], on='FromNodeId')

qmul_papers_in_degree.dropna(inplace=True)

qmul_papers_in_degree.columns = ['citations', 'Paper']

qmul_papers_in_degree[['Paper', 'citations']]


# ## Save Dataframes

# In[38]:


# Save dataframes
write(f"{DATASETS_FOLDER}/parquet/paper_citations.pq", citations)
write(f"{DATASETS_FOLDER}/parquet/abstracts.pq", abstracts)
write(f"{DATASETS_FOLDER}/parquet/tld_aggregation.pq", tld_df)
write(f"{DATASETS_FOLDER}/parquet/domain_aggregation.pq", domain_df)


# In[ ]:




