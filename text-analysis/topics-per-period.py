#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re

import numpy as np
import pandas as pd

# Text analysis
import gensim
from gensim import corpora
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

import spacy

# Visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_colwidth", 200)
pyLDAvis.enable_notebook()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# function to plot most frequent terms
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top n most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
    
    return d
    


# In[3]:


DATASETS_FOLDER = "datasets"
ABSTRACTS_FOLDER = "abstracts"

ABSTRACTS_FOLDER_PATH = f"{DATASETS_FOLDER}/{ABSTRACTS_FOLDER}/"


# In[4]:


with open(f"../{DATASETS_FOLDER}/abstracts_with_last_citation.csv", 'r') as f:
    df = pd.read_csv(f)

df['text'] = df['title'] + ' ' + df['description']
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))

df['date'] = df['date'].astype('datetime64[ns]')
# df['date_from'] = df['date_from'].astype('datetime64[ns]')
df['last_citation'] = df['last_citation'].astype('datetime64[ns]')


# In[5]:


df.head()


# In[6]:


# Initialising NLTK components
stop_words = stopwords.words('english')
stop_words.append('the')

# NLTK Stemming and Lemmatizer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
wordnet_lemmatizer = WordNetLemmatizer()


# In[7]:


# Text processing functions

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

def preprocess_texts(texts):
    papers_text = []
    
    for t in texts:
        pt = []

        # Using spacy lemmatisation
        # doc = nlp(t) 
        # papers_text.append(" ".join([token.lemma_ for token in doc if len(token) > 2]))

        for w in t.split():
            if len(w) > 2:
                # Using stemming
                # pt.append(stemmer.stem(w))
                # Using lemmatisation
                pt.append(wordnet_lemmatizer.lemmatize(w))
        papers_text.append(" ".join(pt))

    # make entire text lowercase
    texts_lower = [r.lower() for r in papers_text]

    # remove stopwords from the text
    output = [remove_stopwords(r.split()) for r in texts_lower]
    
    return output


# In[8]:


periods = [
    {'start_date': '1991-01-01', 'end_date': '1993-12-31'}, # Including 1991 as there is a paper the 1991-12-31
    {'start_date': '1994-01-01', 'end_date': '1995-12-31'},
    {'start_date': '1996-01-01', 'end_date': '1997-12-31'},
    {'start_date': '1998-01-01', 'end_date': '1999-12-31'},
    {'start_date': '2000-01-01', 'end_date': '2001-12-31'},
    {'start_date': '2002-01-01', 'end_date': '2004-12-31'} # Including the papers up to 2004-04-30
]


# In[9]:


for period in periods:
    start_date = period['start_date']
    end_date = period['end_date']

    print(f'Period {start_date} to {end_date}')
    
    papers_in_period = df[((start_date <= df['date']) & (df['date'] <= end_date))]

    print(f'Total papers to process {papers_in_period.shape[0]} in period')

    text_in_period = list(papers_in_period.text)
    texts = preprocess_texts(text_in_period)
    
    print("Most frequent words: ", freq_words(texts, 20))

    tokenized_text = [x.split() for x in texts]
    print("Sample of tokenised text: ", tokenized_text[0], tokenized_text[1])

    dictionary = corpora.Dictionary(tokenized_text)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_text]

    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=3, random_state=100,
                    chunksize=1000, passes=10)

    print('LDA topics: ', lda_model.print_topics())

    # Prepare visualisation and output results
    vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
    
    output_lda_visualisation_file = f'lda_{start_date}_{end_date}.html'
    pyLDAvis.save_html(vis, output_lda_visualisation_file)
    
    print(f'Saved output LDA visualisation in {output_lda_visualisation_file}')

    print('========================')


# In[13]:





# In[ ]:




