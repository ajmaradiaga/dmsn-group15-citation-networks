#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx


# In[2]:


get_ipython().system('head -n 10 datasets/cit-HepTh-no-cycles.csv')


# In[3]:


G = nx.DiGraph()


# In[4]:


with open('datasets/cit-HepTh-no-cycles.csv', 'r') as f:
    i = 0
    for l in f:
        if not l.startswith('#') and i != 0:
            # Leaving them as ints as some might start with 00
            G.add_edge(*l.strip().split(','))
        i += 1


# In[5]:


G['9309119']
#, '9305047', '9311130', '9303159'


# In[6]:


G['9305047']


# In[7]:


G['9303159']


# In[8]:


G['9311130']


# In[9]:


G['9209052'], G['9205062']


# In[10]:


G.number_of_edges(), G.number_of_nodes()


# In[11]:


nx.average_clustering(G)


# In[12]:


nx.transitivity(G)


# In[13]:


nx.is_weakly_connected(G)


# In[14]:


nx.is_strongly_connected(G)


# In[15]:


scc_count = 0

for scc in nx.strongly_connected_components(G):
    scc_count += 1
    
scc_count


# In[16]:


wcc_count = 0

for wcc in nx.weakly_connected_components(G):
    wcc_count += 1
    print(len(wcc))
    if wcc_count > 1000:
        break
    

wcc_count


# In[17]:


nx.is_connected(G)


# In[ ]:


nx.triangles(G)


# In[ ]:




