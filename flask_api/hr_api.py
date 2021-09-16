# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:20:20 2021

@author: ouss3ma
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)




def hr_prediction(df):
    
    #encoding features
    df['enrolled_university'] = enrol.transform(df['enrolled_university'])
    df['relevent_experience'] = exp.transform(df['relevent_experience'])
    df['education_level'] = edu.transform(df['education_level'])
    df['Major_discipline'] = maj.transform(df['Major_discipline'])
    df['company_type'] = typ.transform(df['company_type'])
    df['gender'] = gen.transform(df['gender'])
    
    #standarize input
    df_std = scaler_std.transform(df)
    
    #prediction probability
    prediction_prob = model.predict_proba(df_std)
    
    return prediction_prob[0][1]

    

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp
    

@app.route("/")
def hello():
    return "Welcome to the HR Analytics: Job Change of Data Scientist Prediction APIs!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_, index = [0])
        
    except Exception as e:
        raise e
        
    if query_df.empty:
        return bad_request()
    
    else:
        prediction = hr_prediction(query_df)    
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    #load classifier
    model = pickle.load(open('models\model.pkl', 'rb'))
    
    #load standarization
    scaler_std = pickle.load(open('models\std.pkl', 'rb'))

    #load label encoders
    exp = pickle.load(open('models\exp.pkl', 'rb'))
    enrol = pickle.load(open('models\enrol.pkl', 'rb'))
    edu = pickle.load(open('models\edu.pkl', 'rb')) 
    maj = pickle.load(open('models\maj.pkl', 'rb'))
    typ = pickle.load(open('models\typ.pkl', 'rb'))
    gen = pickle.load(open('models\gen.pkl', 'rb'))
    
    print('load done!!')
    
    app.run(debug=True)
    
    