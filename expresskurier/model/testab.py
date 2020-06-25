import json
import numpy as np
import pandas as pd
import joblib
import os.path
import requests
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score

def load_data(filename, features, target, company):
    hour_divisor = 6
    df = pd.read_json(filename, lines=True)
    X = df.loc[:, features].values

    Y = df.loc[:, target].values.ravel()

    return X, Y

url = 'http://127.0.0.1:8000/predict/'

companies = [360, 516, 620]
for company in companies:
    features = ['city', 'shipment_day', 'hour']
    target = ['delta_time']
    filename = f"data/{company}_testingA.jsonl"
    XA, YA = load_data(
        filename, features, target, company)

    filename = f"data/{company}_testingB.jsonl"
    XB, YB = load_data(
        filename, features, target, company)
    with open(f'{company}_test_ab.log', 'w') as f:
        for x in XA:
            payload = dict(
                city=x[0],
                shipment_day=x[1],
                hour=x[2],
            )
            r = requests.post(url, json=payload)

        for x in XB:
            payload = dict(
                city=x[0],
                shipment_day=x[1],
                hour=x[2],
            )
            r = requests.post(url, json=payload)

    
# tutaj mialobyc sprawdzenie z MSE ale braklo czasu :(
