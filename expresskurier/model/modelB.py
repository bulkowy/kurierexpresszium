import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from util import *

version = 'B'

def training(company: int):
    features = ['city', 'shipment_day', 'hour']
    target = ['delta_time']
    ohe, need_fit = get_OHE(version)

    filename = f'data/{company}_training.jsonl'
    Xt, Yt = load_data(
        filename, features, target, ohe, 
        version=(version if need_fit else None))
    
    filename = f'data/{company}_validation.jsonl'
    Xv, Yv = load_data(
        filename, features, target, ohe)
        
    filename = f'data/{company}_testingB.jsonl'
    Xtst, Ytst = load_data(
        filename, features, target, ohe)
    
    model = KNeighborsRegressor()
    model.fit(Xt, Yt)

    train = create_measures(Yt,model.predict(Xt))
    val = create_measures(Yv,model.predict(Xv))
    oot = create_measures(Ytst,model.predict(Xtst)) 
    print_measures(train, val, oot, 'PRE')

    params = get_best_params(model, Xv, Yv, version, company)
    
    model2 = KNeighborsRegressor(**params)
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
