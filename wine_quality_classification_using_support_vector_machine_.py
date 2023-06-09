# -*- coding: utf-8 -*-
"""Wine quality classification using Support Vector Machine .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ErxEmp_lX_l30dKLthRCQuqzGPSO3llb

Title: Classifying the data using the Support Vector Machine.
"""

#Import the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

"""Load the Dataset"""

df = pd.read_csv("winequality-red.csv")
df.head()

#check the data
df.info()

df.nunique()

"""Distribution of classes"""

good_q= df[df['residual sugar']==2.0][0:200]
bad_q= df[df['residual sugar']==9.0][0:200]


    
# plotting the bubble chart
axes = good_q.plot(kind='scatter', x="pH", y="total sulfur dioxide", color='red', label='good quality')
bad_q.plot(kind='scatter', x="pH", y="total sulfur dioxide", color='blue', label='bad quality', ax=axes)
  


# showing the plot

"""Identify unwanted rows"""

df.dtypes

#There are no unwanted columns

"""Split data into Train/Test"""

x= df.drop("quality", axis=1)
y=df['quality']

x=x.values
y=y.values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x, y, test_size =0.2, random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

"""Modeling (SVM using scikitlearn)"""

from sklearn import svm

classifier=svm.SVC(kernel ='linear', gamma='auto', C=1.5)
#fit the model
classifier.fit(x_train, y_train)
y_predict=classifier.predict(x_test)




#print the first 30 true and predicted responses

"""Evaluation(Results)"""

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

"""Saving and Loading the model"""

#save the existing model to file
import pickle

filename='THE WINE_QUALITY_MODEL.sav'
pickle.dump(classifier, open(filename,"wb"))

#Load a saved model
loaded_model = pickle.load(open("THE WINE_QUALITY_MODEL.sav" ,"rb"))

#Make predictions

input_data = (5.6,1.30,0.006,12.5,21.8,2.6,5.9,0.9,1.5,16.8,5.9)

#changing the input_data to numpy array
input_data_as_numpy_array =np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

if (prediction >= 5):
    print('The wine is of a good quality')
else:
    print('The wine is of a bad quality')