#!/usr/bin/env python
# coding: utf-8

# In[6]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import numpy
import os
import sys


# In[2]:


# lod weights into new model
loaded_model = load_model("model1.h5")


# In[3]:


# load the dataset
datanames = sys.argv[1]
dataset = loadtxt(datanames, delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# In[4]:


# make class predictions with the model
predictions = loaded_model.predict_classes(X)


# In[5]:


# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# In[ ]:




