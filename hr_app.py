# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:19:44 2021

@author: ouss3ma
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
         # HR Analytics: Job Change of Data Scientist
         
         This app predicts if a data scientist will leave his currecnt job
         Data optained from Kaggle

         """)
         
st.sidebar.header('User Input Features')

def user_input():
    city_dev_idx = st.sidebar.slider('City developement index', 0.0, 1.0, 0.5)
    gender = st.sidebar.selectbox('Gender', ('male', 'female', 'other'))
    relevent_experience = st.sidebar.selectbox('Relevent experience', 
                    ('Has relevent experience', 'No relevent experience'))
    enrolled_university = st.sidebar.selectbox('enrolled university', 
                    ('No enrollement', 'Full time course', 'Part time course'))
    education_level = st.sidebar.selectbox('education_level', 
                    ('Graduate', 'Masters', 'High school', 'Phd', 
                     'Primary school'))
    Major_discipline = st.sidebar.selectbox('Major discipline', 
                    ('STEM', 'Business Degree', 'Arts', 'Humanities', 
                     'Other', 'No major'))
    experience = st.sidebar.slider('experience', 0, 21, 10)
    campany_size = st.sidebar.selectbox('campany size', 
                    ('<10', '10-49', '50-99', '100-499', '500-999', 
                     '1000-4999', '5000-9999', '>10000'))
    campany_type = st.sidebar.selectbox('campany type', 
                    ('Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 
                     'Public Sector', 'NGO', 'Other'))
    last_new_job = st.sidebar.slider('last new job', 0, 5, 0)
    Training_hours = st.sidebar.slider('Training hours', 0, 500, 100)
    
    data = {'city_dev_idx' : city_dev_idx,
            'gender' : gender,
            'relevent_experience' : relevent_experience,
            'enrolled_university' : enrolled_university,
            'education_level' : education_level,
            'Major_discipline' : Major_discipline,
            'experience' : experience,
            'campany_size' : campany_size,
            'campany_type' : campany_type,
            'last_new_job' : last_new_job,
            'Training_hours' : Training_hours
        }
    
    features = pd.DataFrame(data, index=[0])
    return  features

df = user_input()

#replace company size intervall with integers

df['company_size'] = df['company_size'].replace('<10', 0)
df['company_size'] = df['company_size'].replace('<10', 0)

df['company_size'] = df['company_size'].replace('10-49', 1)
df['company_size'] = df['company_size'].replace('10-49', 1)

df['company_size'] = df['company_size'].replace('50-99', 2)
df['company_size'] = df['company_size'].replace('50-99', 2)

df['company_size'] = df['company_size'].replace('100-499', 3)
df['company_size'] = df['company_size'].replace('100-499', 3)

df['company_size'] = df['company_size'].replace('500-999', 4)
df['company_size'] = df['company_size'].replace('500-999', 4)

df['company_size'] = df['company_size'].replace('1000-4999', 5)
df['company_size'] = df['company_size'].replace('1000-4999', 5)

df['company_size'] = df['company_size'].replace('5000-9999', 6)
df['company_size'] = df['company_size'].replace('5000-9999', 6)

df['company_size'] = df['company_size'].replace('>10000', 7)
df['company_size'] = df['company_size'].replace('>10000', 7)




#load label encoders
exp = pickle.load(open('exp.pkl', 'rb'))
enrol = pickle.load(open('enrol.pkl', 'rb'))
edu = pickle.load(open('enrol.pkl', 'rb')) 
maj = pickle.load(open('maj.pkl', 'rb'))
typ = pickle.load(open('typ.pkl', 'rb'))
gen = pickle.load(open('gen.pkl', 'rb'))


#encoding features

df['relevent_experience'] = exp.transform(df['relevent_experience'])

df['enrolled_university'] = enrol.transform(df['enrolled_university'])

df['education_level'] = edu.transform(df['education_level'])

df['Major_discipline'] = maj.transform(df['Major_discipline'])

df['campany_type'] = typ.transform(df['campany_type'])

df['gender'] = gen.transform(df['gender'])


#display user input features
st.subheader('user Input Features') 
st.write(df)
  


#load classifier


#predictions


#display predictions
st.subheader('Prediction')
output = np.array(['will leave', 'will stay'])
#st.write(output[prediction])

st.subheader('Prediction Probability')
#st.write(prediction_proba)