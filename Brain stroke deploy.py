

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:48:32 2023

@author: Fresh
"""

import numpy as np
import pickle
import streamlit as st

loaded_model= pickle.load(open(r"C:\Users\Fresh\Desktop\streamlit\lr1_logistic_Regression_model_stroke_pkl.sav",'rb'))

#creating a function for prediction

def Brain_stroke(input_data):
    #Make predictions

    

    #changing the input_data to numpy array
    input_data_as_numpy_array =np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction == 0:
        return 'The patient does not have a Brain stroke'
    else:
        return 'The patient may have a Brain stroke'
    
    
    
def main():
    # give a title for the interface
    st.title('Brain stroke classification WebApp')
    
    #getting the input data from the user
    
    
    
    age= st.text_input('Age of patient in years')
    hypertension= st.text_input('hypertension')
    heart_disease= st.text_input('heart disease')
    avg_glucose_level= st.text_input('Average glucose level')
    bmi= st.text_input('Body mass index')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Show test result'):
        diagnosis= Brain_stroke([age, hypertension, heart_disease, avg_glucose_level, bmi])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__ =='__main__':
    main()
    
    
    