import json
import numpy as np
import pandas as pd
import joblib
import os.path
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

def create_measures(y,y_pred): 
    MSE_test = mean_squared_error(y, y_pred)
    MAE_test = mean_absolute_error(y, y_pred)
    R2_test = r2_score(y, y_pred)
    
    d = {
        'MSE': [round(MSE_test,4)], 
        'MAE': [round(MAE_test,4)],
        'R2': [round(R2_test,4)]
    }
    d = pd.DataFrame.from_dict(d)
    return d

def load_data(filename, features, target, ohe, version, company):
    hour_divisor = 6
    df = pd.read_json(filename, lines=True)
    X = df.loc[:, features]
    if 'hour' in X:
        X['hour'] = X['hour'].floordiv(hour_divisor)

    if not os.path.exists(f'{version}_{company}_ohe.joblib'):
        ohe.fit(X)
        joblib.dump(ohe, f'{version}_{company}_ohe.joblib')

    X = ohe.transform(X)
    Y = df.loc[:, target].values.ravel()

    return X, Y

def save_validation_set(filenames, ohe, Xv, Yv, Y_):
    for filename in filenames:
        with open(filename, "w+") as result:
            for xv, yv, y_ in zip(Xv,Yv,Y_):
                result.write(json.dumps({
                    "city": ohe.inverse_transform(xv.reshape(1, -1))[0][0], 
                    "delivery_time": float(yv), 
                    "predicted_time": float(y_)
                }))
                result.write("\n")

def create_paramgrid():
    C = [0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    epsilon = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    gamma = [0.0001, 0.002, 0.0005, 0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]

    hyperparameters = dict(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
    )

    return hyperparameters

def dump_best_params(params: dict, version: str, company: int):
    filename = f'{version}_{company}_best.params'
    with open(filename, 'w') as f:
        for k, v in params.items():
            f.write(f'Best {k}: {v}\n')

def get_best_params(model, Xv, Yv, version, company):
    hyperparameters = create_paramgrid()
    clf = GridSearchCV(model, hyperparameters, cv=5, scoring='neg_mean_squared_error', n_jobs=4)
    best_model = clf.fit(Xv, Yv)
    params = best_model.best_estimator_.get_params()
    dump_best_params(params, version, company)

    return params

def print_measures(train, val, oot, prefix):
    print(f'{prefix} GRIDSEARCH _____________')
    print('TRAIN')
    print(train)
    print('VAL')
    print(val)
    print('TEST')
    print(oot)