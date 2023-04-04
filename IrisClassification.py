#!/usr/bin/env python
# coding: utf-8
Import the necessary libraries
# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# Load the Iris Dataset

# In[19]:


iris = pd.read_csv('D:/iris/Iris.csv')


# In[20]:


iris.head


# In[21]:


iris.describe()


# In[22]:


iris.info()


# In[23]:


iris["Species"].value_counts()


# In[26]:


sns.FacetGrid(iris,hue="Species",height=5).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend()


# In[35]:


x=iris[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
y=iris[["Species"]].values


# In[36]:


Model=LogisticRegression()
Model.fit(x,y)


# In[37]:


#accurarcy
Model.score(x,y).round(2)


# In[38]:


#prediction
Actual=y
predicted=Model.predict(x)


# In[39]:


from sklearn import metrics
print(metrics.classification_report(Actual,predicted))


# In[40]:


print(metrics.confusion_matrix(Actual,predicted))


# In[41]:


predicted=Model.predict([[5.1,3.5,1.4,0.2]])
predicted


# In[ ]:




