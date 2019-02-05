
# coding: utf-8

# In[1]:


import networkx as nx


# In[2]:


get_ipython().system(u'head -n 10 datasets/cit-HepTh-processed.csv')


# In[3]:


G = nx.DiGraph()


# In[4]:


with open('datasets/cit-HepTh-processed.csv', 'r') as f:
    i = 0
    for l in f:
        if not l.startswith('#'):
            # Leaving them as ints as some might start with 00
            G.add_edge(*l.split())
            i += 1
            if i >= 1000:
                break


# In[5]:


nx.write_gexf(G, 'gephi/cit-HepTh-1000.gexf')


# In[6]:


nx.draw(G)


# In[7]:


get_ipython().magic(u'pinfo2 G.add_edge')

