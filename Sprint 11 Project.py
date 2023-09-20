#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# The project is accepted! Good luck on the final sprint!
# 
# </div>

# **Review**
# 
# Hi, my name is Dmitry and I will be reviewing your project.
#   
# You can find my comments in colored markdown cells:
#   
# <div class="alert alert-success">
#   If everything is done successfully.
# </div>
#   
# <div class="alert alert-warning">
#   If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
#   
# <div class="alert alert-danger">
#   If a section requires some corrections. Work can't be accepted with red comments.
# </div>
#   
# Please don't remove my comments, as it will make further review iterations much harder for me.
#   
# Feel free to reply to my comments or ask questions using the following template:
#   
# <div class="alert alert-info">
#   For your comments and questions.
# </div>
#    
# First of all, thank you for turning in the project! You did a great job overall, there is only one small problem that needs to be fixed before the project is accepted. Let me know if you have questions!

# Mobile carrier Megaline has found out that many of their subscribers use legacy plans. They want us to develop a model that would analyze subscribers' behavior and recommend one of Megaline's newer plans: Smart or Ultra.
# 
# We have access to behavior data about subscribers who have already switched to the new plans (from the project for the Statistical Data Analysis course). For this classification task, we need to develop a model that will pick the right plan. 
# 
# Let's develop a model with the highest possible accuracy. In this project, the threshold for accuracy is 0.75. We'll check the accuracy using the test dataset.

# In[1]:


import pandas as pd
from sklearn import set_config
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('/datasets/users_behavior.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.isna().sum() #check for any missing fields


# In[6]:


df.duplicated().sum() #check for duplicates


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected
# 
# </div>

# In[7]:


#lets create a train and validation set. This will output the df_train at 80% and df_valid at 20%
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=12345) 


# In[8]:


#lets create a test set taking 25% of df_train (which is at 80%), to give df_test a total of 20% of the original dataframe.
#which makes df_train now at 60%
df_train, df_test = train_test_split(df_train, test_size=0.25, random_state=12345)


# In[9]:


features_train = df_train.drop(['is_ultra'], axis=1)


# In[10]:


target_train = df_train['is_ultra']


# In[11]:


features_valid = df_valid.drop(['is_ultra'], axis=1)


# In[12]:


target_valid = df_valid['is_ultra']


# In[13]:


features_test = df_test.drop(['is_ultra'], axis=1)


# In[14]:


target_test = df_test['is_ultra']


# In[15]:


print(features_train.shape)
print(target_train.shape)
print(features_valid.shape)
print(target_valid.shape)
print(features_test.shape)
print(target_test.shape)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data split is reasonable
# 
# </div>

# In[16]:


for depth in range(1, 11): # choose hyperparameter range 10
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth) # set max_depth=depth
    model.fit(features_train, target_train) # train model on training set
    predictions_valid = model.predict(features_valid) #predict based off validation set
    print('max_depth =', depth, ': ', end='')
    print(accuracy_score(target_valid, predictions_valid)*100)


# When using the DecisionTreeClassifier model the best depth out of 10 is 5 & 7 which have an accuracy score of 78.85%.

# In[17]:


for depth in range(1, 51): # choose hyperparameter range 50
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth) # set max_depth=depth
    model.fit(features_train, target_train) # train model on training set
    predictions_valid = model.predict(features_valid) #predict based off validation set
    print('max_depth =', depth, ': ', end='')
    print(accuracy_score(target_valid, predictions_valid)*100)


# When using the DecisionTreeClassifier model the best depth out of 50 is 5 & 7 which have an accuracy score of 78.85%. One thing to note is that starting at depth 28 and onward the accuracy score is the same.

# In[18]:


best_score = 0
best_est = 0
for est in range(1, 11): # choose hyperparameter range
    model = RandomForestClassifier(random_state=12345, n_estimators=est) # set number of trees
    model.fit(features_train, target_train) # train model on training set
    score = model.score(features_valid, target_valid) # calculate accuracy score on validation set
    if score > best_score:
        best_score = score # save best accuracy score on validation set
        best_est = est # save number of estimators corresponding to best accuracy score

print("Accuracy of the best model on the test set (n_estimators = {}): {}".format(best_est, best_score*100))

final_model = RandomForestClassifier(random_state=12345, n_estimators=10) # change n_estimators to get best model
final_model.fit(features_train, target_train)


# When using the RandomForestClassifier with 10 estimators the best one is 10 at 78.69%.

# In[19]:


best_score = 0
best_est = 0
for est in range(1, 51): # choose hyperparameter range
    model = RandomForestClassifier(random_state=12345, n_estimators=est) # set number of trees
    model.fit(features_train, target_train) # train model on training set
    score = model.score(features_valid, target_valid) # calculate accuracy score on validation set
    if score > best_score:
        best_score = score # save best accuracy score on test set
        best_est = est # save number of estimators corresponding to best accuracy score

print("Accuracy of the best model on the test set (n_estimators = {}): {}".format(best_est, best_score*100))

final_model = RandomForestClassifier(random_state=12345, n_estimators=50) # change n_estimators to get best model
final_model.fit(features_train, target_train)


# When using the RandomForestClassifier with 50 estimators the best one is 44 at 79.16%.

# In[20]:


model = LogisticRegression(random_state=12345, solver='liblinear') # initialize logistic regression constructor with parameters random_state=12345 and solver='liblinear'
model.fit(features_train, target_train)  # train model on training set
score_train = model.score(features_train, target_train)*100 # calculate accuracy score on training set 
score_valid = model.score(features_valid, target_valid)*100 # calculate accuracy score on test set  

print("Accuracy of the logistic regression model on the training set:", score_train,)
print("Accuracy of the logistic regression model on the test set:", score_valid,)


# <div class="alert alert-danger">
# <s><b>Reviewer's comment</b>
# 
# Great, you tried a couple of different models and did some hyperparameter tuning. One small problem: after you're done with model selection, you need to evaluate the final model on the holdout set (not used for training/hyperparameter tuning/model selection) to get an unbiased estimate of its generalization performance. That was the point of splitting the data into three parts, and not two :)
#     
# Usually the validation set would be used for hyperparameter tuning/model selection, and the test set for final model evaluation, but the names don't really matter
# 
# </div>

# <div class="alert alert-info">
#     Just so I understand properly. I am to train/tune the model with the training and validation sets, then I am to run the module with test set separately? Just want to clarify. 

# <div class="alert alert-danger">
# <s><b>Reviewer's comment V2</b>
# 
# Yep, so e.g. after we're done comparing different models, we find out that the best model is random forest with n_estimators = 44, then we just fit it using the train set and calculate accuracy score on the test set.
#     
# By the way, as the validation set is not needed after we're done with hyperparameter tuning/model selection, we can refit the final model using the combined train+validation data
# 
# </div>

# <div class="alert alert-info">
#     So would that just be the original df set beore I used the train_test_split or do I need to go back and rename the first df_train to something else so it retains the 80%? Because right now df_train is at 60% and if I add df_train and df_valid I would get 80%.
#     Below I have it set up to use the 60% df_train, is this what you're looking for?

# <div class="alert alert-success">
# <b>Reviewer's comment V3</b>
# 
# Yeah, sure, this is fine!
#     
# > So would that just be the original df set beore I used the train_test_split or do I need to go back and rename the first df_train to something else so it retains the 80%?
#     
# Yep, you could rename the `df_train` in the first split into something like `df_train_valid` or leave the splitting as is and concatenate them afterwards:
#     
# ```python
# df_train_valid = pd.concat([df_train, df_valid])
# ```
# 
# </div>

# In[21]:


model = RandomForestClassifier(random_state=12345, n_estimators=44)
model.fit(features_train, target_train)
score = model.score(features_test, target_test)
print(score*100)


# Using the LogisticRegression Model the accuracy of the test comes out to 72.94%. So this model will not work.

# The best model is the RandomForestClassifier as it gives us the highest accuracy out of 50 estimators is 44 at 79.16% and out of 10 is 5 & 7 estimators at 78.85%. - when using the validation set.
# 
# When using the test set, for the best model and estimators the accuracy is at 79.47%.
# 
# One thing we have to keep in mind is that DecisionTree has lower accuracy because it can become overfitted or underfitted depending on the amount of tree depth. Random Forset will always have the highest accuracy because it uses an ensemble of trees instead of just one. The logistic regression model, is straightforward so there will be no overfitting.
# 
# However, Logisitic regression is the fastest one because it has the least number of model parameters. The speed of decision tree is also high and it depends on the depth of the tree. A random forest is the slowest: the more trees there are, the slower this model works.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Great!
# 
# </div>
