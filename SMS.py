#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


# In[2]:


sms = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 22 - Naive Bayes\Datasets\sms_raw_NB.csv",encoding='ISO-8859-1')
sms.head(10)


# In[3]:


#Time to clean the data from unrequired data and removing stop, filler word


# In[5]:


stop_words=[]
with open(r"C:\Users\srira\Desktop\Ram\Data science\Prerequisites\stopwords.txt") as sw:
    stop_words=sw.read()
stop_words = stop_words.split("\n")    


# In[17]:


def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    w = [w for w in w if w not in stop_words]
    return " ".join(w)


# In[18]:


sms.text = sms.text.apply(cleaning_text)


# In[19]:


sms.text = sms.loc[sms.text!=" ",:]


# In[21]:


sms_train,sms_test = train_test_split(sms,test_size=0.3)


# In[46]:


sms_test.shape


# In[22]:


#It's time for count vectorizer


# In[23]:


#First we need to define function to split into words


# In[24]:


def split_into_words(i):
    return [word for word in i.split(" ")]


# In[26]:


sms_count_vec = CountVectorizer(split_into_words).fit(sms.text)


# In[28]:


all_sms = sms_count_vec.transform(sms.text)
train_sms = sms_count_vec.transform(sms_train.text)
test_sms = sms_count_vec.transform(sms_test.text)


# In[ ]:


#Using tfidf transformer


# In[29]:


tfidf = TfidfTransformer().fit(all_sms)
train_tfidf = tfidf.transform(train_sms)
test_tfidf = tfidf.transform(test_sms)


# In[35]:


#Now we can build model


# In[32]:


mnb = MultinomialNB()
gnb = GaussianNB()


# In[33]:


mnb.fit(train_tfidf,sms_train.type)


# In[34]:


pred_train = mnb.predict(train_tfidf)
pred_test = mnb.predict(test_tfidf)


# In[39]:


Acc_train = np.mean(pred_train==sms_train.type)
Acc_test = np.mean(pred_test==sms_test.type)
Acc_train,Acc_test


# In[40]:


#gaussian NB


# In[42]:


gnb.fit(train_tfidf.toarray(),sms_train.type.values)


# In[44]:


pred_train = gnb.predict(train_tfidf.toarray())
pred_test = gnb.predict(test_tfidf.toarray())


# In[45]:


Acc_train = np.mean(pred_train==sms_train.type)
Acc_test = np.mean(pred_test==sms_test.type)
Acc_train,Acc_test

