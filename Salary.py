#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# In[3]:


Salary = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 22 - Naive Bayes\Datasets\SalaryData_Train.csv")
Salary.head(10)


# In[4]:


Salary_test = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 22 - Naive Bayes\Datasets\SalaryData_Test.csv")
Salary_test.head(10)


# In[5]:


#Data has complete labels, so we may need to use label encoder to encode to numeric categorical


# In[6]:


string_col = ['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[8]:


for i in string_col:
    encoder = LabelEncoder()
    Salary[i]=encoder.fit_transform(Salary[i])
    Salary_test[i]=encoder.fit_transform(Salary_test[i])


# In[9]:


#Building model


# In[10]:


x=Salary.drop(['Salary'],axis=1)
y=Salary['Salary']
x_test= Salary_test.drop(['Salary'],axis=1)
y_test=Salary_test['Salary']


# In[12]:


gnb = GaussianNB()
mnb = MultinomialNB()


# In[13]:


gnb.fit(x,y)


# In[14]:


pred_train = gnb.predict(x)
pred_test = gnb.predict(x_test)


# In[16]:


#Accuracy


# In[17]:


confusion_matrix(pred_train,y)


# In[19]:


pd.crosstab(y.values,pred_train)


# In[20]:


Acc_train = np.mean(y==pred_train)
Acc_test = np.mean(y_test==pred_test)
Acc_train,Acc_test


# In[21]:


#Let's build multinomial model


# In[22]:


mnb.fit(x,y)


# In[23]:


pred_train = mnb.predict(x)
pred_test = mnb.predict(x_test)


# In[24]:


confusion_matrix(pred_train,y)
pd.crosstab(y.values,pred_train)


# In[25]:


Acc_train = np.mean(y==pred_train)
Acc_test = np.mean(y_test==pred_test)
Acc_train,Acc_test


# In[ ]:


#Hence Gausian model is far better. Since most the data is categorical i assume this is the best accuracy that we can acheive

