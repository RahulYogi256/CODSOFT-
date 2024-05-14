#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.read_csv("advertising.csv")
df.head()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


sns.pairplot(df , x_vars=['TV','Radio','Newspaper'],y_vars="Sales" ,kind="scatter")
plt.show()


# In[13]:


df['TV'].plot.hist(bins=10)


# In[14]:


df['Radio'].plot.hist(bins=10,color="red")


# In[15]:


df['Newspaper'].plot.hist(bins=10,color="green")


# In[16]:


sns.heatmap(df.corr(),annot = True)
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df[["TV"]],df[["Sales"]], test_size = 0.3,random_state=0)


# In[21]:


print(X_train)


# In[22]:


print(Y_train)


# In[23]:


print(Y_test)


# In[24]:


print(X_test)


# In[26]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)


# In[28]:


res= model.predict(X_test)
print(res)


# In[29]:


model.coef_


# In[32]:


model.intercept_


# In[33]:


0.05473199* 69.2 + 7.14382225


# In[34]:


plt.plot(res)


# In[36]:


plt.scatter(X_test,Y_test)
plt.plot(X_test,7.14382225 +0.05473199 * X_test,'r')
plt.show()


# In[ ]:





# In[ ]:




