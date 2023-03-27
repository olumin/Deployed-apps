# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:32:59 2023

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

loaded_model= pickle.load(open(r'C:\Users\Fresh\Desktop\streamlit\THE WINE_QUALITY_MODEL.pkl','rb'))

#creating a function for prediction

def Wine_quality_pred(input_data):
    #Make predictions

    

    #changing the input_data to numpy array
    input_data_as_numpy_array =np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction >= 5):
        return('The wine is of a good quality')
    else:
        return('The wine is of a bad quality')
    
    
    
def main():
    # give a title for the interface
    st.title('Wine quality prediction WebApp')
    
    #getting the input data from the user
    
    
    
    fixed_acidity= st.text_input('Fixed acidity')
    volatile_acidity= st.text_input('Volatile acidity')
    citric_acid= st.text_input('Chest pain type')
    residual_sugar= st.text_input('residual sugar')
    chlorides= st.text_input('chlorides')
    free_sulfur_dioxide= st.text_input('free sulfur dioxide')
    total_sulfur_dioxide= st.text_input('total sulfur dioxide')
    density= st.text_input('density')
    pH= st.text_input('pH')
    sulphates= st.text_input('sulphates')
    alcohol= st.text_input('alcohol')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Show result'):
        diagnosis= Wine_quality_pred([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide, density,pH, sulphates, alcohol])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__ =='__main__':
    main()
    
    
    