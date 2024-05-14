#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt


# In[29]:


titanic_data = pd.read_csv('Titanic-Dataset.csv')
titanic_data.head() 


# In[31]:


titanic_data.shape # will give the counts of rows and columns


# In[32]:


titanic_data['Survived'].value_counts()


# In[33]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_data['Survived'].value_counts().keys()),list(titanic_data['Survived'].value_counts()),color=["r","g"])
plt.show


# In[34]:


titanic_data['Pclass'].value_counts()


# In[35]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_data['Pclass'].value_counts().keys()),list(titanic_data['Pclass'].value_counts()),color=["Blue","Red","yellow"])
plt.show


# In[36]:


titanic_data['Sex'].value_counts()


# In[37]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_data['Sex'].value_counts().keys()),list(titanic_data['Sex'].value_counts()),color=["blue","pink"])
plt.show


# In[38]:


plt.figure(figsize=(5,7))
plt.hist(titanic_data['Age'])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.show()


# In[39]:


titanic_data['Survived'].isnull()


# In[40]:


sum(titanic_data['Survived'].isnull())


# In[42]:


sum(titanic_data['Age'].isnull())


# In[44]:


titanic_data=titanic_data.dropna()


# In[46]:


titanic_data['Age'].isnull()


# In[47]:


sum(titanic_data['Age'].isnull())


# In[49]:


x_train=titanic_data[['Age']]
y_train=titanic_data[['Survived']]


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[54]:


dtc = DecisionTreeClassifier()


# In[55]:


dtc.fit(x_train,y_train)


# In[56]:


y_pred = dtc.predict(x_train)


# In[57]:


y_pred


# In[ ]:




