#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing and Reading the dataset

# In[2]:


df = pd.read_csv(r"E:\heart disease analysis LR.csv")
df.head()


# # Analysis of Data

# In[3]:


df.shape


# In[4]:


df.keys()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# ### Removing Nan /NULL values from the data

# In[8]:


df.dropna(axis = 0, inplace =True)
print(df.shape)


# In[9]:


df['TenYearCHD'].value_counts()


# # Data Visualization

# ### Correlation Matrix

# In[12]:


plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='Purples' , annot = True, linecolor='Green', linewidths=1.0)
plt.show()


# ### Pairplot

# In[13]:


sns.pairplot(df)
plt.show()


# ### Count plot of people based on their sex and whether they are current smoker or not

# In[15]:


sns.catplot(data = df, kind='count', x= 'male', hue='currentSmoker')
plt.show()


# ### Countplot - subplots of No. of people affecting with ChD on basis of their sex and current smoking

# In[17]:


sns.catplot(data=df, kind= 'count', x= 'TenYearCHD', col = 'male' , row= 'currentSmoker', palette = 'Blues')
plt.show()


# # Machine Learning Part

# ### Separating the data into feature and target data.

# In[20]:


x = df.iloc[:,0:15]
y = df.iloc[:,15:16]


# In[21]:


x.head()


# In[22]:


y.head()


# ### Importing the model and assigning the data for  training and test set

# In[48]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state =21)


# # Applying the ML model - Logestic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# # Training the data

# In[50]:


logreg.fit(x_train, y_train)


# ### Testing the data

# In[51]:


y_pred = logreg.predict(x_test)


# ### Predicting the score

# In[52]:


score = logreg.score(x_test,y_test)
print("Prediction score is :",score)


# # Getting the confusion Matrix and Classification Report
# 

# ### Confusion Matrix
# 

# In[53]:



from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix is:\n",cm)


# ### Classification Report

# In[54]:


print("Classification Report is :\n\n", classification_report(y_test,y_pred))


# ### Plotting the confusion matrix

# In[58]:


conf_matrix = pd.DataFrame(data = cm,
                          columns = ['Predicted :0', 'Predicted:1'],
                          index = ['Actual:0', 'Actual:1'])
plt.figure(figsize = (10,6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor = "Black", linewidths = 1.5)
plt.show()


# In[ ]:




