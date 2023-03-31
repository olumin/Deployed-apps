#!/usr/bin/env python
# coding: utf-8

# ##  ML model that optimizes to predict patient with heart disease using the RandomForest, Linear SVC classifier.
# 

# **Target Parameter Optimization: Precision, Accuracy**

# ##### Problem type: Classification
# ##### Models
# ##### A. RandomForest
# ##### B. Linear SVC

#  **Methodology**
# 
# 1. Getting the data ready
# 2. Choose the right estimator(model)/algorithm for the problem
# 3. Model/algorithm fitting to make predictions on data
# 4. Evaluating the model
# 5. Improve the model
# 6. Save and load the trained model
# 7. Integrate all*
#Keywords: 
#AUC: Area Under the Curve
#ROC: Receiver operating characteristics
#Fpr-False Positive rate
#Tpr-True Positive rate**
# #### 1. Getting the data ready
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
heart_disease= pd.read_csv("heart-disease.csv")
heart_disease


# In[2]:


#Create a (feature matrix) x
x = heart_disease.drop("target", axis=1)
y= heart_disease["target"]

print(x)


# In[3]:


#sc = MinMaxScaler(feature_range=(0,1))
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


# In[4]:


y


# #### 2. Choose the right model and hyperparameters 
# 

# 
# 
# #### A. RandomForestClassifier

# In[5]:


import sklearn
sklearn.show_versions()


# In[6]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=100)

#We'll keep the default hyperparameters
clf.get_params()


# #### 3. Model Fitting

# In[7]:


#Split the data into training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2
                           
                                               )


# In[8]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[9]:


clf.fit(x_train, y_train)


# #### 5. Evaluating the RandomForest classifier  
#a) Metric functions-Confusion Matrix
#b) Scoring parameters
# #### a) Confusion Matrix

# In[10]:


clf.score(x_train, y_train)


# In[11]:


clf.score(x_test, y_test)


# In[12]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_preds = clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[13]:


confusion_matrix(y_test, y_preds)


# In[14]:


accuracy_score(y_test, y_preds)

#Comment: The Confusion matrix summary shows the same result (83.6%). The classification is accurate
# #### 6. Model improvement

# In[15]:


#Try different amount of n_estimators

np.random.seed(42)
for i in range(10,100,10):
    print(f"Trying model with {i} estimators...")
    clf= RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) * 100:.2f}%")
    print("")


# #### 7.Save the model and load it

# In[16]:


import pickle
pickle.dump(clf, open("random_forest_model_1.pk1", "wb"))


# In[17]:


loaded_model= pickle.load(open("random_forest_model_1.pk1", "rb"))
loaded_model.score(x_test, y_test)


#   Comment: The Random forest classifier model score shows 85% which is fair enough, but this can as well be improved. We try another estimator below

# ### Show Predictions

# In[18]:


y_preds=clf.predict(x_test)
y_preds[:10]


# In[19]:


np.array(y_test[:10])


# **Compare the predictions to the truth**

# In[20]:



from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_preds)

#The mean absolute error for the predicted value or its likely deviation from the original value is +_0.20.. which quite workable 
# #### b) Scoring Parameter

# **Evaluating metrics**
#  i. accuracy
# ii. precision
# iii.recall
# iv. f1

# In[21]:


from sklearn.model_selection import cross_val_score
np.random.seed(42)
cv_scp= cross_val_score(clf, x,y, cv=5, scoring =None)
cv_scp
# cross-validated accuracy
print(f"The cross-validated accuracy is: {np.mean(cv_scp)*100:.2f}%")


# In[22]:


# i. cross-validated accuracy
np.random.seed(42)
cv_scp= cross_val_score(clf, x,y, cv=5, scoring ="accuracy")
print(f"The cross-validated accuracy is: {np.mean(cv_scp)*100:.2f}%")


# In[23]:


# ii. Precision
cv_precision= cross_val_score(clf, x,y, cv=5, scoring ="precision")
np.mean(cv_precision)


# In[24]:


#iii. Recall
cv_recall= cross_val_score(clf, x,y, cv=5, scoring ="recall")
np.mean(cv_recall)


# In[25]:


#iv. f1
cv_f1= cross_val_score(clf, x,y, cv=5, scoring ="f1")
np.mean(cv_f1)


# ### Option2: Using another estimator for classification 

# ### B. Linear SVC

# sklearn link: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
#Consulting the map to try linear SVC
# In[26]:


#import the linearsvc estimator class


# In[27]:


from sklearn.svm import LinearSVC

#setup random seed
np.random.seed(42)
#Make the data
x= heart_disease.drop("target", axis=1)
y= heart_disease["target"]

#split the data

x_train, x_test, y_train,y_test= train_test_split(x,y, test_size=0.2)
#Instatiate linear SVC
clf1=LinearSVC(max_iter=1000)

#fit the model to the data (train the machine learning model)
clf1.fit(x_train,y_train)

#Evaluate the LinearSVC(patterns from learned data)
clf1.score(x_test, y_test)


# In[28]:


heart_disease["target"].value_counts()

#Comment: LinearSVC seems better with the score of 86.9%
# **Make the Predictions**
# 
# #Let's make predictions on the test data

# In[29]:


clf1.predict(x_test)


# In[30]:


np.array(y_test)


# # Compare the predictions to the truth labels to evaluate the model

# In[31]:


y_preds=clf1.predict(x_test)
np.mean(y_preds == y_test)


# In[32]:


clf1.score(x_test, y_test)


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)


# #### Evaluating the Linear SVC model
# 
# 
# 
# ​

# **Evaluation Metrics**
# 
# 1. **Score Parameters**:
#      'accuracy',
#      'average precision',
#      'recall',
#      'f1' 
# 2. **Metric functions**
#     AUC/ROC score ,  '
#     Confusion matrix,
#     Classification report
# 

# ##### 1. Source Parameter- cross_validation
# 

# # On the Linear SVC

# In[34]:


from sklearn.model_selection import cross_val_score
np.random.seed(42)
clf1=LinearSVC(max_iter=1000)

#let's parse the scoring parameter

cross_val_score(clf1, x, y, cv=5, scoring=None)


# # On the RandomForest

# In[35]:


cross_val_score(clf, x, y, cv=5, scoring=None)


# 

# **i. Accuracy**

# # On the Linear SVC

# In[36]:


#i. Accuracy
clf_cross_val_score1=np.mean(cross_val_score(clf1, x, y, cv=5, scoring="accuracy"))
clf_cross_val_score1


# # On the RandomForest

# In[37]:


clf_cross_val_score1b=np.mean(cross_val_score(clf, x, y, cv=5, scoring="accuracy"))
accuracy=clf_cross_val_score1b
accuracy


# # On the Linear SVC

# **ii.precision**

# In[38]:


#ii. precision
clf_cross_val_score2=np.mean(cross_val_score(clf1, x, y, cv=5, scoring="precision"))
clf_cross_val_score2


# In[ ]:





# In[39]:


# On the RandomForest


# In[40]:


#ii. precision
clf_cross_val_score2=np.mean(cross_val_score(clf, x, y, cv=5, scoring="precision"))
precision=clf_cross_val_score2
precision


# **iii.Recall**

# # On the Linear SVC

# In[41]:


#iii. Recall
cross_val_score3=cross_val_score(clf1, x, y, cv=5, scoring="recall" )
np.mean(cross_val_score3)


# # On the RandomForest

# In[42]:


#iii. Recall
cross_val_score3=cross_val_score(clf, x, y, cv=5, scoring="recall" )
Recall=np.mean(cross_val_score3)
Recall


# **iv.f1**

# # On the Linear SVC

# In[43]:


#iv. f1
clf_cross_val_score4=np.mean(cross_val_score(clf1, x, y, cv=5, scoring="f1"))
clf_cross_val_score4


# # On the RandomForest

# In[44]:


#iv. f1
clf_cross_val_score4=np.mean(cross_val_score(clf, x, y, cv=5, scoring="f1"))
f1=clf_cross_val_score4
f1


# In[45]:


print(f"Heart Disease Classifier Cross-validated Accuracy Linear SVC: {(clf_cross_val_score1 * 100):.2f}%")
print(f"Heart Disease Classifier Cross-validated Accuracy RandomForest:{(clf_cross_val_score1b * 100):.2f}%")


# Comments: From several iterations,the Linear SVC model have about 60-70% likelihood of predicting the right label i.e about 6-7 out of 10 times while the RandomForest have over 80% likelihood

# In[46]:


Dict1=[{"Accuracy": accuracy, "Precision":precision,"Recall":Recall,"F1":f1}]
cross_val_before_improvement= pd.DataFrame(Dict1)
cross_val_before_improvement


# #### RandomForestClassifier evaluation using AUC/ROC

# #### 2. Metric functions

# **i. AUC/ROC**
#comparison of a model true positive rate vs model false positive rate
#*True positive= model predicts 1 when truth is 1 
#*False positive= model predicts 1 when truth is 0
#*True negative =model predicts 0 when truth is 0
#*False negative =model predicts 0 when truth is 1
# In[47]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=100)


# In[48]:


#Split the data into training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2
                           
                                               )


# In[49]:


clf.fit(x_train,y_train)


# In[50]:


from sklearn.metrics import roc_curve

#make predictions with probabilities
y_prob = clf.predict_proba (x_test)

y_prob[:10], len(y_prob)


# In[51]:


y_prob_positive = y_prob[:,1]
y_prob_positive[:10]


# In[52]:


#Calculate fpr, tpr, thresholds
fpr, tpr, thresholds=roc_curve(y_test, y_prob_positive)

#check the false positive rates
fpr


# In[53]:


tpr


# In[54]:


thresholds


# In[55]:


#Create a function for plotting ROC Curve

import matplotlib.pyplot as plt
def plot_roc_curve(fpr,tpr):
    """
    plots a ROC curve given the fpr and tpr of the model
    """
    #Plot Roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    #Plot line with no predictive power (baseline)
    plt.plot([0,1], [0,1], color='darkblue', linestyle="--", label="Guessing")
    
    #Customize the plot
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title('Receiver Operating Characteristics (ROC) curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr, tpr)
    


# In[56]:


from sklearn.metrics import roc_auc_score
   
roc_auc_score(y_test, y_prob_positive)


# In[57]:


#plot a perfect ROC curve and AUC score
fpr, tpr, thresholds= roc_curve(y_test,y_test)
plot_roc_curve (fpr,tpr)


# In[58]:


#perfect AUC score
roc_auc_score(y_test, y_test)


# ##### ii. Confusion Matrix
# 
#The confusion matrix will help to compare the labels a model predicts and the actual labels it was meant to predict. It reveals where the model is getting confused.
# In[59]:


from sklearn.metrics import confusion_matrix
y_preds= clf.predict(x_test)
confusion_matrix(y_test, y_preds)


# In[60]:


#visualize the matrix with pandas.crosstab()
pd.crosstab(y_test, y_preds, rownames=["Actual labels"],
            colnames= ["Predicted Labels"])


# In[61]:


21+4+6+30


# In[62]:


len(x_test)


# In[63]:


#Make the confusion matrix more visual with heatmap

import seaborn as sns
#set the font scale
sns.set(font_scale=1.5)
#create a confusion matrix
con_mat= confusion_matrix(y_test,y_preds)
#plot it using seaborn
sns.heatmap(con_mat, annot=True,fmt=".1f",linewidth=.1)
plt.xlabel("True label")
plt.ylabel("predicted label");


# **iii. Classification Report**

# In[64]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


#  Comment: visit the sklearn module webpage to understand the headings
#  Link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#A perfect model has an f1-score of 1.00. f1-score of 0.86 is good, perfect accuracy is equal to 1.0, we recorded an accuracy of 0.84

#*Tips: Accuracy is a good measure to start with if all classes are balanced (e.g same amount of samples which are labelled with 0 or 1
#*Precision and recall become more important when classes are imbalanced
#*if false positive are worse tan false negatives, aim for higher precision
#*If false negative predictions are worse than false positives, aim for higher recall.
#*F1-score is a combination of precision and recall
# ### 5.Improving the Model

# **Hyperparameter tuning**
#Three ways to adjust hyperparameters
#1. By hand
#2. Randomly wth RandomSearchCV
#3. Exhaustively with GridSearchCVwe adjust the hyperparameters to improve the models By Hand#let's adjust:
#'max_depth'
#'max_features'
#'min_samples_leaf'
#'min_samples_split'
#'n_estimators' 
# In[65]:


clf.get_params()


# In[66]:


np.random.seed(42)
#shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)
#split into x &y
x= heart_disease_shuffled.drop("target", axis=1)
y= heart_disease_shuffled["target"]

#split the data into train, validation &test sets
train_split = round(0.7) *len(heart_disease_shuffled) #70% of data
valid_split=round(train_split + 0.15 * len(heart_disease_shuffled)) #15% of data
x_train, y_train= x[:train_split],y[:train_split]
x_valid, y_valid =x[train_split:valid_split:], y[train_split:valid_split]
x_test,y_test =x[valid_split:], y[:valid_split]
len(x_train), len(x_valid), len(x_test)

#Hyper parameter tuning by Hand did not work; showing 0 samples in the length of the split, we need a minimum of 1 for each split length to adjust n_estimator,maxdepth

# ##### Hyperparameter tuning with GridSearchCV on the RandomForestClassifier

# In[67]:


from sklearn.model_selection import GridSearchCV
#create a dictionary with the hyperparameters we'll like to adjust
grid = {"n_estimators": [100,200],
         "max_depth": [None],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [6],
        "min_samples_leaf": [1,2]}


# In[68]:


#split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2)
# instantiate RandomForestClassifier
clf=RandomForestClassifier(n_jobs=None)
#setup GridsearchCV
rs_clf = GridSearchCV(estimator=clf, param_grid=grid, #number of models to try 
                                cv=5, verbose=2)
    #Fit the GridsearchCV version of clf
rs_clf.fit(x_train, y_train);


# In[69]:


rs_clf.best_params_


# In[70]:


#Let's now use the best parameter to train the model
from sklearn.model_selection import GridSearchCV
#create a dictionary with the hyperparameters we'll like to adjust
grid = {"n_estimators": [200],
        "max_depth": [None],
        "max_features": ["sqrt"],
        "min_samples_split": [6],
        "min_samples_leaf": [2]}


# In[71]:


#split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2)
# instantiate RandomForestClassifier
rs_clf=RandomForestClassifier(n_jobs=None)
#setup GridsearchCV
rs_clf = GridSearchCV(estimator=rs_clf, param_grid=grid, #number of models to try 
                                cv=5, verbose=2)
    #Fit the GridsearchCV version of clf
rs_clf.fit(x_train, y_train);


# In[72]:


#Create an evaluation function
def evaluate_preds(y_true, rs_y_preds):
    
    accuracy=accuracy_score(y_true, rs_y_preds)
    precision=precision_score(y_true, rs_y_preds)
    recall=recall_score(y_true, rs_y_preds)
    
    metric_dict ={"accuracy": round(accuracy, 2),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1": round(f1, 2)}
   
    
    
    
    print(f"Acc: {accuracy*100:.2f}%")
    print(f"precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict


#   **Compare our diffrent models metrics**

# In[73]:




Dict2 = [{'Accuracy': accuracy, "Precision": precision,"Recall":Recall,  "F1":f1}]
cross_val_after_improvement=pd.DataFrame(Dict2)
cross_val_after_improvement


# In[74]:



cross_val_before_improvement.plot.bar(figsize=(10,8));

plt.title("cross val before model improvement")

cross_val_after_improvement.plot.bar(figsize=(10,8));

plt.title("cross val after model improvement")

#Comments: Comparing the baseline model with the improved, we find out that the Accuracy, precision, Recall,f1 metrics is fully optimized .The model is a good one for deployment.
# In[ ]:





# #### 6. Saving and Loading the trained models
# 1. with python's pickle module
# 2. joblib
# 
# 
# **pickle**

# In[75]:


clf=RandomForestClassifier()
clf.fit(x_test, y_preds)
import pickle


#save an existing model to file
pickle.dump(clf, open("Rs_random_Random_Forest_model.pkl","wb"))
pickle.dump(sc, open("Random_scaled.pkl", "wb"))


# In[86]:


#Load a saved model
loaded_model = pickle.load(open(r"C:\Users\Fresh\Desktop\streamlit\Rs_random_Random_Forest_model.pkl","rb"))
loaded_scaled_model = pickle.load(open("Random_scaled.pkl","rb"))


# **Make some predictions with the model**

# In[87]:


#Make predictions

input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3)

#changing the input_data to numpy array
input_data_as_numpy_array =np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The patient does not have a heart disese')
else:
    print('The patient may have a heart disease')







# In[ ]:




