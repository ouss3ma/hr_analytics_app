# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 21:33:47 2021

@author: ouss3ma
"""
from flask import Blueprint, Flask, request, jsonify

hello_api = Blueprint('hello_api', __name__) 



@hello_api.route("/")
def hello():
    return "Welcome to the HR Analytics: Job Change of Data Scientist Prediction APIs!"