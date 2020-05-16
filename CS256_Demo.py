#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd
import sklearn.metrics as metrics
from joblib import dump, load


# ## Load data

# In[2]:


data = pd.read_csv('Data/creditcard_testing.csv')
X = data.drop(columns = ["Class"])
y = data["Class"]


# ## Function to print the scores

# In[3]:


def printScores(model_name):
    clf = load(model_name)
    pred = clf.predict(X)
    
    # Originally, -1 is the outlier for LOF output
    # Swap the label so that 1 is the outlier for LOF
    if("LOF" in model_name):
        pred[pred==-1] = 0
        pred[pred==1] = -1
        pred[pred==0] = 1

    print(f"accuracy: {metrics.accuracy_score(y, pred)}")
    print(f"precision: {metrics.precision_score(y, pred)}")
    print(f"recall: {metrics.recall_score(y, pred)}")
    print(f"f1_score: {metrics.f1_score(y, pred)}")
    
    


# ## Load pre-trained model and test

# In[4]:


classifiers = ["LOF", "SVM-rbf", "SVM-poly"]    


# ### Original without resampling

# In[5]:


print("Before resampling:")
for clf_name in classifiers:
    print(clf_name)
    filename = "SavedModels/" + clf_name + ".joblib"
    printScores(filename)


# ### with resampling

# In[6]:


print("After resampling:")
for clf_name in classifiers:
    print(clf_name)
    filename = "SavedModels/" + clf_name + "_re.joblib"
    printScores(filename)

