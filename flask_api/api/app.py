# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:20:20 2021

@author: ouss3ma
"""

from flask import Flask, request, jsonify
from .endpoints.predict import predict_api 
from .endpoints.hello import hello_api 


app = Flask(__name__)
app.register_blueprint(predict_api) 
app.register_blueprint(hello_api) 


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
    