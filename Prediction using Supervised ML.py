#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt


# In[3]:


url="http://bit.ly/w-data"
data_load = pd.read_csv(url)  
print("Successfully imported data into console" )


# In[4]:


data_load.head(10)


# In[7]:


data_load.plot(x='Hours', y='Scores', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Scored')    
plt.show()  


# In[8]:


X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values 


# In[9]:


from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training ... Completed !.")  


# In[11]:


line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show() 


# In[12]:


print(X_test)   
y_pred = regressor.predict(X_test) 


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df


# In[14]:


hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0]))  


# In[ ]:




