#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn.metrics as metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from joblib import dump, load


# ## Data Overview

# In[2]:


df = pd.read_csv('Data/creditcard.csv')
print(f"Original: {df.shape}")

print(f"Null values? {df.isnull().values.any()}")
df = df.drop_duplicates(subset=None, keep='first', inplace=False)
print(f"After removing depulicates: {df.shape}")

df.describe()


# ## Scale Time and Amount

# In[3]:


scaled_time = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))
scaled_amount = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df.insert(0, 'scaled_time', scaled_time)
df.insert(1, 'scaled_amount', scaled_amount)

data = df.drop(['Time','Amount'], axis=1)
data.describe()


# ## Change label from {0, 1} to {-1,+1}

# In[4]:


data['Class'].replace(0, -1, inplace=True)


# ## Check labels

# In[5]:


fraud = data[data['Class']==1]
normal = data[data['Class']==-1]
print(f"num of fraud: {len(fraud)}; num of normal: {len(normal)}")

labels ='fraud', 'normal'
size = [len(fraud), len(normal)]
 
# Create a circle for the center of the plot
my_circle = plt.Circle((0,0), 0.5, color='white')
plt.pie(size, labels=labels, colors=['yellow','skyblue'], autopct='%1.2f%%', startangle=90)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# ## Set up for training
# ### Set up 5-fold cross validation

# In[6]:


X = data.drop(columns = ['Class'])
y = data['Class']
cv = KFold(shuffle=True)


# ### Set classifiers

# In[7]:


classifiers = {    
    "LOF" : LocalOutlierFactor(n_neighbors=20, novelty = True),
    "SVM-rbf" : SVC(),  
    "SVM-poly" : SVC(kernel="poly")
}


# ### Set score names

# In[8]:


score_names = ["time","accuracy","precision","recall","f1"]


# ###  Set a function to get the scores

# In[9]:


def getScores(classifier_name, train_x, train_y, test_x, test_y):
    classifier = classifiers[classifier_name]
    start = time.time()
    classifier = classifier.fit(train_x, train_y)
    end = time.time()
    exe_time = end - start
    
    pred = classifier.predict(test_x)
    
    # Originally, -1 is the outlier for LOF output
    # Swap the label so that 1 is the outlier for LOF
    if(classifier_name=="LOF"):
        pred[pred==-1] = 0
        pred[pred==1] = -1
        pred[pred==0] = 1
       
    acc = metrics.accuracy_score(test_y, pred)
    pre = metrics.precision_score(test_y, pred, zero_division=1)
    rec = metrics.recall_score(test_y, pred, zero_division=1)
    f1 = metrics.f1_score(test_y, pred, zero_division=1)
    return [exe_time, acc, pre, rec, f1]


# ### Set a function to get average score of cross validation

# In[10]:


def avgScore(the_results):
    the_sums = [0]*len(the_results[0])
    for r in the_results:
        for i in range(len(r)):
            the_sums[i]+=r[i]
    
    return [the_sum/len(the_results) for the_sum in the_sums] 


# ## Training
# ### Get 5-folds for cross validation and resampling

# In[11]:


folds_origin = []
folds_re1 = []
folds_re2 = []
folds_re3 = []
times = []
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    print(f"Number of fraud in test case: {len(y_test[y_test==1])}")
    folds_origin.append([X_train, X_test, y_train, y_test])
    
    #resampling 1: under + SMOTE, final ratio 0.5:
    under = RandomUnderSampler(sampling_strategy=0.0026)
    X_train_re, y_train_re = under.fit_resample(X_train, y_train)
    start1 = time.time()
    over = SMOTE(sampling_strategy=0.5)
    X_train_re1, y_train_re1 = over.fit_resample(X_train_re, y_train_re)
    end1 = time.time()
    exe_time1 = end1 - start1
    folds_re1.append([X_train_re1, X_test, y_train_re1, y_test])
    
    #resampling 2: under + SMOTE + ENN, final ratio 0.5:
    start2 = time.time()
    enn = EditedNearestNeighbours(sampling_strategy='all')
    X_train_re2, y_train_re2 = enn.fit_resample(X_train_re1, y_train_re1)
    end2 = time.time()
    exe_time2 = end2 - start2 + exe_time1
    folds_re2.append([X_train_re2, X_test, y_train_re2, y_test])

    #resampling 3: under + SMOTE + ENN, final ratio 1:
    start3 = time.time()
    over = SMOTEENN()
    X_train_re3, y_train_re3 = over.fit_resample(X_train_re, y_train_re)
    end3 = time.time()
    exe_time3 = end3 - start3
    folds_re3.append([X_train_re3, X_test, y_train_re3, y_test])
    
    times.append([exe_time1, exe_time2, exe_time3])
    
# store the folds for original and different re-samplings
folds = {
    "Original" : folds_origin,
    "SMOTE0.5" : folds_re1,
    "SMOTEENN0.5" : folds_re2,
    "SMOTEENN" : folds_re3
}

# print average time for original and different re-samplings
i = 0
for fold in folds:
    if(fold!="Original"):
        print(f"{fold} avg exe time: {avgScore(times)[i]}")
        i+=1


# ### Training models and get result

# In[12]:


for clf_name in classifiers:
    print(f"\n{clf_name}", end='\t')
    
    avgs = []
    # for each data set
    for fold_name in folds:
        print(fold_name, end='\t')
        fold = folds[fold_name]
        results =[]
        # train each fold
        for f in fold:
            X_train, X_test, y_train, y_test = f
            results.append(getScores(clf_name, X_train, y_train, X_test, y_test))
        # get average scores
        avgs.append(avgScore(results))
       
    # print result
    for i in range(len(score_names)):
        print(f"\n{score_names[i]}", end='\t')
        for avg in avgs:
            print(f"{avg[i]}",end='\t') 


# ## Save the models
# ### Save the testing data

# In[13]:


X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.3)
test_data = pd.concat([X_testing, y_testing], axis=1, sort=False)
test_data.to_csv(r'Data/creditcard_testing.csv', index = False)


# ### Original without re-sampling

# In[14]:


for clf_name in classifiers:
        filename = "SavedModels/" + clf_name + ".joblib"
        clf = classifiers[clf_name] 
        clf.fit(X_training, y_training)
        dump(clf, filename)
        print(f"{filename} saved.")


# ### With re-sampling

# In[15]:


under_s = RandomUnderSampler(sampling_strategy=0.0026)
X_training_re, y_training_re = under_s.fit_resample(X_training, y_training)
over_s = SMOTEENN(sampling_strategy=0.5)
X_training_re, y_training_re = over_s.fit_resample(X_training_re, y_training_re)

for clf_name in classifiers:
        filename = "SavedModels/" + clf_name + "_re.joblib"
        clf = classifiers[clf_name] 
        clf.fit(X_training_re, y_training_re)
        dump(clf, filename)
        print(f"{filename} saved.")

