import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from util import *

version = 'A'


def training(company: int):
    features = ['city']
    target = ['delta_time']
    ohe = OneHotEncoder(sparse=False)

    filename = f"data/{company}_training.jsonl"
    Xt, Yt = load_data(
        filename, features, target, ohe, version, company)
    
    filename = f"data/{company}_validation.jsonl"
    Xv, Yv = load_data(
        filename, features, target, ohe, version, company)
    
    filename = f"data/{company}_testingA.jsonl"
    XA, YA = load_data(
        filename, features, target, ohe, version, company)

    filename = f"data/{company}_testingB.jsonl"
    XB, YB = load_data(
        filename, features, target, ohe, version, company)

    Xtst = np.concatenate((XA, XB))
    Ytst = np.concatenate((YA, YB))

    model = SVR(kernel='rbf')
    model.fit(Xt, Yt)

    train = create_measures(Yt,model.predict(Xt))
    val = create_measures(Yv,model.predict(Xv))
    oot = create_measures(Ytst,model.predict(Xtst)) 
    print_measures(train, val, oot, 'PRE')

    params = get_best_params(model, Xv, Yv, version, company)
    
    model2 = SVR(kernel='rbf', C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'])
    model2.fit(Xt, Yt)
    Y_ = model2.predict(Xv)

    train = create_measures(Yt,model2.predict(Xt))
    val = create_measures(Yv,model2.predict(Xv))
    oot = create_measures(Ytst,model2.predict(Xtst)) 
    print_measures(train, val, oot, 'POST')

    filenames = [f"{version}_{company}best.jsonl"]
    save_validation_set(filenames, ohe, Xv, Yv, Y_)

    filenames = [f"{version}_{company}_best.joblib"]
    for filename in filenames:
        joblib.dump(model2, filename)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-c",
        "--company",
        action = "store",
        choices = [360, 516, 620],
        type = int,
        required = True        
    )
    program_args = vars(argument_parser.parse_args())
    training(program_args["company"])
