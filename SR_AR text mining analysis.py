#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
import re
import sys
import datetime  


# In[ ]:





# # Import data

# In[1]:


start = datetime.datetime.now()
list_of_files = os.listdir("/Users/yuanp/Documents/SR_AR_2010-2018txt")
nlist = len(list_of_files)
count_data = list(np.arange(0,nlist,1))
train = pd.DataFrame(np.zeros((nlist, 2)), index=count_data, columns=['Name','Text'])

for file, i in zip(list_of_files, count_data):
    f = open("/Users/yuanp/Documents/SR_AR_2010-2018txt" + "/" + file, "r")
    train.loc[i, "Name"] = (file[:-4])
    train.loc[i, "Text"] = f.read()

train.head()


# In[3]:


# train['Ticker'], train['Kode'], train['Year'] = train['Name'].str.split('_', 2).str


# In[4]:


# train.head()


# In[5]:


#train['Text']=train['Text'].str.replace('\n',' ')
#train['Name']=train['Name'].str.replace('_SR','SR')
#train['Name']=train['Name'].str.replace('_',' ')


# In[6]:


#train.tail()


# In[501]:


# train.to_csv("E:/tm/project2/data_mentah.csv",index = True)


# In[10]:


len(train)


# In[11]:


train


# ## 1.1 Calculate number of words

# In[12]:


train['Jumlah Kata'] = train['Text'].apply(lambda x: len(str(x).split(" ")))
train.head()


# ## 1.2 Calculate number of characters

# In[13]:


train['Jumlah Karakter'] = train['Text'].str.len() ## this also includes spaces
train[['Text','Jumlah Karakter']].head()


# ## 1.3 Calculate the average of word length

# In[14]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['Rata2 panjang kata'] = train['Text'].apply(lambda x: avg_word(x))
train[['Text','Rata2 panjang kata']].head()


# ## 1.4 Calculate number of unimportant words

# In[15]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

train['Kata2 tidak penting'] = train['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['Text','Kata2 tidak penting']].head()


# ## 1.5 Calculate number of special characters

# In[16]:


train['hastags'] = train['Text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['Text','hastags']].head()


# ## 1.6 Calculate number of numerics

# In[17]:


train['Numerik'] = train['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['Text','Numerik']].head()


# ## 1.7 Calculate number of uppercase words

# In[18]:


train['upper'] = train['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['Text','upper']].head()


# ## 2. Basic Pre-processing

# ### 2.1 Lower case
# ### 2.2 Removing Punctuation

# In[19]:


# Lower case
train['Text'] = train['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Remove punctuation
train['Text'] = train['Text'].str.replace('[^\w\s]','')

# Remove numerics
train['Text'] = train['Text'].apply(lambda x: ' '.join([x for x in x.split() if not x.isdigit()]))

# Remove indonesian stopwords
from nltk.corpus import stopwords
stop = stopwords.words('indonesian')
train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Remove english stopwords
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(('pt','tbk','laporan','report','company','ii','i','perusahaan','p','perseroan','jl','indonesia','persero'))
train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train['Text'].head()


# # 3. Calculate number of positive and negative words

# In[20]:


pw = pd.read_fwf("E:/tm/project2/sentiment dictionary/positive-words.txt")
nw = pd.read_fwf("E:/tm/project2/sentiment dictionary/negative-words.txt")
train['Positif'] = train['Text'].apply(lambda x: len([x for x in x.split() if x in pw["words"].tolist()]))
train['Negatif'] = train['Text'].apply(lambda x: len([x for x in x.split() if x in nw["words"].tolist()]))
train[['Text','Positif','Negatif']].head(10)


# # 4. Retrieve popular words each company/sic in sort

# In[21]:


freq = []
i = 0
nfreq = 0
nama_perusahaan = []
i = 0
for c in train["Name"]:
    nama_perusahaan.append(c)
    freq.append(pd.Series(''.join(train.iloc[i,1]).split()).value_counts())
    nfreq = len(freq[i].keys()) if nfreq < len(freq[i].keys()) else nfreq
    i+=1

freq_data = list(np.arange(0,nfreq,1))
freqtable = pd.DataFrame(np.zeros((nfreq, len(freq))), index=freq_data, columns=nama_perusahaan)

for i in zip(freq_data):
    j = 0
    for perusahaan in nama_perusahaan:
        nWord = len(freq[j].keys())
        
        freqtable.loc[i, perusahaan] = freq[j].keys()[i] if i[0] < nWord else '-'
        j+=1
freqtable.head(5)


# In[22]:


freqtable.to_excel("E:/tm/project2/rank_word_ec.xlsx",index=False)


# # 5. Retrieve popular words of documents

# In[23]:


most_words = pd.Series(' '.join(train['Text']).split()).value_counts()
most_words = pd.DataFrame(most_words.keys(),columns=["Kata populer"])
most_words.to_csv("E:/tm/project2/most_wordall.csv",index = True)


# # 6. Sentiment analysis

# In[24]:


import textblob
from textblob import TextBlob


# In[25]:


train['sentiment'] = train['Text'].apply(lambda x: TextBlob(x).sentiment[0] )


# In[600]:


# trainnew['Ticker'], trainnew['Kode'], trainnew['Year'] = trainnew['Name'].str.split('_', 2).str


# In[601]:


# pw = pd.read_fwf("E:/tm/project2/sentiment dictionary/positive-words.txt")
# nw = pd.read_fwf("E:/tm/project2/sentiment dictionary/negative-words.txt")
# trainnew['Positif'] = trainnew['Text'].apply(lambda x: len([x for x in x.split() if x in pw["words"].tolist()]))
# trainnew['Negatif'] = trainnew['Text'].apply(lambda x: len([x for x in x.split() if x in nw["words"].tolist()]))


# In[26]:


train.head()


# In[616]:


# new = train[['Ticker','Kode','Year','Positif','Negatif','sentiment']]


# In[617]:


# new.to_excel("E:/tm/project2/hasil.xlsx",index=False)


# In[27]:


train['sentiment'] = train['Text'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['Text','sentiment']].head(10)


# In[28]:


train.to_excel("E:/tm/project2/summarize.xlsx",index=False)


# # 7. Calculate number words each category

# In[29]:


neg_word = nw # Negative
ne_word = neg_word.values.tolist()
rank_neg = train.copy()

i = 0
for month in neg_word['words']:
    colname = month
    rank_neg[colname] = rank_neg['Text'].apply(lambda x: len([x for x in x.split() if x in ne_word[i]]))
    i+=1


# In[30]:


rank_neg = rank_neg.drop(rank_neg.iloc[:,1:12] ,  axis='columns')
rank_neg.set_index(['Name'], inplace = True)
rank_neg = rank_neg.transpose()
rank_neg.to_csv("E:/tm/project2/rank_neg.csv",index=True)


# In[31]:


pos_word = pw # positive
po_word = pos_word.values.tolist()
rank_pos = train.copy()

i = 0
for month in pos_word['words']:
    colname = month
    rank_pos[colname] = rank_pos['Text'].apply(lambda x: len([x for x in x.split() if x in po_word[i]]))
    i+=1


# In[32]:


rank_pos = rank_pos.drop(rank_pos.iloc[:,1:12] ,  axis='columns')
rank_pos.set_index(['Name'], inplace = True)
rank_pos = rank_pos.transpose()
rank_pos.to_csv("E:/tm/project2/rank_pos.csv",index = True)


# # 8. System Recomendation

# In[33]:


import os
from os import path
import numpy as np
import graphlab
import pandas as pd
import matplotlib.pyplot as plt
from graphlab import SFrame
import string
import nltk
import re
import sys


# In[37]:


data = graphlab.SFrame(train[['Name','Text']])
data['Word_count'] = graphlab.text_analytics.count_words(data['Text'])
tfidf = graphlab.text_analytics.tf_idf(data['Word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

data['tfidf'] = tfidf
knn_model = graphlab.nearest_neighbors.create(data,features=['tfidf'],label='Name')

train["Name"] = train["Name"].astype(int)
a = train["Name"][0]
c = len(train["Name"])
for i in range (0,c):
    a = train["Name"][i]
    a = a.astype('S')
    print "HASIL RUN DATA",a
    industri = data[data['Name'] == a]
    knn_model.query(industri, k=c).print_rows(num_rows=c, num_columns=3)


# In[38]:


finish = datetime.datetime.now()
print "Start Running: ",start
print "End Running: ",finish
print "Time Running: ",int((finish - start).total_seconds())/3600,"Jam"


# # 8. Create wordcloud

# In[ ]:


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
alice_mask = np.array(Image.open(path.join(d, "cloud.png")))
stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", mask=alice_mask, max_words=20000,contour_width=5,contour_color='steelblue',
    stopwords=stopwords)
i = 0
for e in train["Name"]:
    d1=train.iloc[i,1]
    i+=1
    wc.generate(d1)
    wc.to_file(path.join("D:\Career\Paper Text Mining\Wordcloud", e+'.png'))


# In[ ]:


most_words = pd.Series(' '.join(train['Text']).split()).value_counts()
most_words

