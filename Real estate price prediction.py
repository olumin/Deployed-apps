# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:34:28 2023

@author: Fresh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:48:32 2023

@author: Fresh
"""

import numpy as np
import pickle
import streamlit as st

loaded_model= pickle.load(open(r'C:\Users\Fresh\Desktop\streamlit\Rs_random_Random_Forest_model.pkl','rb'))

#creating a function for prediction

def heart_disease_pred(input_data):
    #Make predictions

    input_data = (4,5,2,4,2,4,5,3,2,33,2,4,3)

    #changing the input_data to numpy array
    input_data_as_numpy_array =np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction == 0:
        return 'The patient does not have a heart disease'
    else:
        return 'The patient has a heart disease'
    
    
    
def main():
    # give a title for the interface
    st.title('Heart disease prediction WebApp')
    
    #getting the input data from the user
    
    
    
    age= st.text_input('Age of patient in years')
    sex= st.text_input('Sex of patient')
    cp= st.text_input('Chest pain type')
    trestbps= st.text_input('Resting blood pressure')
    chol= st.text_input('cholesterol')
    fbs= st.text_input('fasting blood sugar')
    restecg= st.text_input('resting electrocardiographic results')
    thalach= st.text_input('Maximum heart rate')
    exang = st.text_input('Exercise induced angina')
    oldpeak= st.text_input('ST depression')
    slope = st.text_input('slope of the peak ST')
    ca= st.text_input('Number of major vessels')
    thal= st.text_input('thalassemia')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Show test result'):
        diagnosis= heart_disease_pred([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__ =='__main__':
    main()
    
    
    