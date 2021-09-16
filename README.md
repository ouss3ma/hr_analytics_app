# HR analytics web app deployed on Heroku

## Objective

This app predicts the probability of a candidate looking for a new job or will work for the company.

## Problem:
         
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. 
Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates. Information related to demographics, education, experience are in hands from candidates signup and enrollment.

## Data

Data obtained from Kaggle:
https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

## App
The deployed web app is live at: https://hr-prediction-app.herokuapp.com/

The web app was built in Python using the following libraries:
* streamlit
* pandas
* numpy
* scikit-learn
* pickle

		 

# Flask API

Exemple of json POST data:

{
    "gender" : "Male",
    "city_dev_idx" : 0.1,
    "relevent_experience" : "Has relevent experience",
    "enrolled_university" : "no_enrollment",
    "education_level" : "Graduate",
    "Major_discipline" : "STEM",
    "experience" : 0,
    "company_size" : 1,
    "company_type" : "Pvt Ltd",
    "last_new_job" : 0,
    "Training_hours" : 0
}