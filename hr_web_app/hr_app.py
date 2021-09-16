# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:19:44 2021

@author: ouss3ma
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
         # HR Analytics: Job Change of Data Scientist
         
         ## Problem
         
         A company which is active in Big Data and Data Science wants to hire 
         data scientists among people who successfully pass some courses which 
         conduct by the company. Many people signup for their training. 
         Company wants to know which of these candidates are really wants 
         to work for the company after training or looking for 
         a new employment because it helps to reduce the cost and time 
         as well as the quality of training or planning the courses and 
         categorization of candidates. Information related to demographics, 
         education, experience are in hands from candidates signup and 
         enrollment.""")

st.write("""
         
         ## Objective
         
         This app predicts the probability of a candidate looking for 
         a new job or will work for the company.
         
         ## Data
         
         Data obtained from Kaggle: 
         https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

         """)
         
st.sidebar.header('User Input Features')

def user_input():
    city_dev_idx = st.sidebar.slider('City developement index', 0.0, 1.0, 0.5)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    relevent_experience = st.sidebar.selectbox('Relevent experience', 
                    ('Has relevent experience', 'No relevent experience'))
    enrolled_university = st.sidebar.selectbox('enrolled university', 
                    ('no_enrollment', 'Full time course', 'Part time course'))
    education_level = st.sidebar.selectbox('education_level', 
                    ('Graduate', 'Masters', 'High School', 'Phd', 
                     'Primary School'))
    Major_discipline = st.sidebar.selectbox('Major discipline', 
                    ('STEM', 'Business Degree', 'Arts', 'Humanities', 
                     'Other', 'No major'))
    experience = st.sidebar.slider('experience', 0, 21, 10)
    company_size = st.sidebar.selectbox('company size', 
                    ('<10', '10-49', '50-99', '100-499', '500-999', 
                     '1000-4999', '5000-9999', '>10000'))
    company_type = st.sidebar.selectbox('company type', 
                    ('Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 
                     'Public Sector', 'NGO', 'Other'))
    last_new_job = st.sidebar.slider('last new job in years', 0, 5, 0)
    Training_hours = st.sidebar.slider('Training hours', 0, 500, 100)
    
    data = {'city_dev_idx' : city_dev_idx,
            'gender' : gender,
            'relevent_experience' : relevent_experience,
            'enrolled_university' : enrolled_university,
            'education_level' : education_level,
            'Major_discipline' : Major_discipline,
            'experience' : experience,
            'company_size' : company_size,
            'company_type' : company_type,
            'last_new_job' : last_new_job,
            'Training_hours' : Training_hours
        }
    
    features = pd.DataFrame(data, index=[0])
    return  features

df = user_input()

#replace company size intervall with integers

df['company_size'] = df['company_size'].replace('<10', 0)

df['company_size'] = df['company_size'].replace('10-49', 1)

df['company_size'] = df['company_size'].replace('50-99', 2)

df['company_size'] = df['company_size'].replace('100-499', 3)

df['company_size'] = df['company_size'].replace('500-999', 4)

df['company_size'] = df['company_size'].replace('1000-4999', 5)

df['company_size'] = df['company_size'].replace('5000-9999', 6)

df['company_size'] = df['company_size'].replace('>10000', 7)




#load label encoders
exp = pickle.load(open('models\exp.pkl', 'rb'))
enrol = pickle.load(open('models\enrol.pkl', 'rb'))
edu = pickle.load(open('models\edu.pkl', 'rb')) 
maj = pickle.load(open('models\maj.pkl', 'rb'))
typ = pickle.load(open('models\typ.pkl', 'rb'))
gen = pickle.load(open('models\gen.pkl', 'rb'))


#encoding features

df['relevent_experience'] = exp.transform(df['relevent_experience'])

df['enrolled_university'] = enrol.transform(df['enrolled_university'])

df['education_level'] = edu.transform(df['education_level'])

df['Major_discipline'] = maj.transform(df['Major_discipline'])

df['company_type'] = typ.transform(df['company_type'])

df['gender'] = gen.transform(df['gender'])


#display user input features
st.subheader('user Input Features') 
st.write(df)
  
#load standarization
scaler_std = pickle.load(open('models\std.pkl', 'rb'))
df_std = scaler_std.transform(df)

#load classifier
model = pickle.load(open('models\model.pkl', 'rb'))


#predictions
#prediction = model.predict(df_std)
prediction_proba = model.predict_proba(df_std)

#display predictions
st.subheader('Prediction')
#output = np.array([
#    'The candidate is probably not looking for job change', 
#    'The candidate is probably looking for job change'])

#st.write(output[int(prediction[0])])

#st.subheader('Prediction Probability')
#st.write(prediction_proba)

#st.write("""the probability of a candidate not looking for a new job or 
#         will not work for the company = """, prediction_proba[0][0])
         
st.write("""the probability of a candidate looking for a new job or 
         will work for the company = """, prediction_proba[0][1])





st.write("""
         ### Github: https://github.com/ouss3ma/hr_analytics_app
         """)