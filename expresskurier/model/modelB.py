import argparse
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import joblib
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

def load_data(filename, features, target):
    df = pd.read_json(filename, lines=True)
    X = df.loc[:, features]

    Y = df.loc[:, target].values
    Y = Y.reshape(Y.size)

    return X, Y

def save_validation_set(filenames, ohe, Xv, Yv, Y_):
    for filename in filenames:
        with open(filename, "w+") as result:
            for xv, yv, y_ in zip(Xv,Yv,Y_):

                result.write(json.dumps({
                    "city": ohe.inverse_transform(xv.reshape(1, -1))[0][1], 
                    "delivery_time": float(yv), 
                    "predicted_time": float(y_)
                }))
                result.write("\n")


def training(company: int):
    features = ['city', 'shipment_day']
    target = ['delta_time']
    ohe = OneHotEncoder(sparse=False)

    filename = f"data/{company}_training.jsonl"
    Xt, Yt = load_data(filename, features, target)
    print(Xt)
    return
    ohe.fit(Xt)
    Xt = ohe.transform(Xt)
    joblib.dump(ohe, '2_ohe.joblib')
    
    filename = f"data/{company}_validation.jsonl"
    Xv, Yv = load_data(filename, features, target)
    Xv = ohe.transform(Xv)
        
    
    filename = f"data/{company}_testing.jsonl"
    Xtst, Ytst = load_data(filename, features, target)
    Xtst = ohe.transform(Xtst)

    C = [0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    epsilon = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    gamma = [0.0001, 0.002, 0.0005, 0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]

    hyperparameters = dict(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
    )
    model = SVR(kernel='rbf')
    model.fit(Xt, Yt)

    test = create_measures(Yt,model.predict(Xt))
    val = create_measures(Yv,model.predict(Xv))
    oot = create_measures(Ytst,model.predict(Xtst)) 

    print("PRE GRIDSEARCH _____________")

    print("TRAIN")
    print(test)
    print("VAL")
    print(val)
    print("TEST")
    print(oot)

    clf = GridSearchCV(model, hyperparameters, cv=5, scoring='neg_mean_squared_error', n_jobs=4)
    best_model = clf.fit(Xv,Yv)

    print("POST GRIDSEARCH _____________")

    params = best_model.best_estimator_.get_params()

    print('Best C:', params['C'])
    print('Best epsilon:', params['epsilon'])
    print('Best gamma:', params['gamma'])

    filebase = ""
    for k, v in params.items():
        filebase += f"{v}_"
    
    model2 = SVR(kernel='rbf', C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'])
    model2.fit(Xt, Yt)

    test = create_measures(Yt,model2.predict(Xt))
    val = create_measures(Yv,model2.predict(Xv))
    oot = create_measures(Ytst,model2.predict(Xtst)) 

    print("TRAIN")
    print(test)
    print("VAL")
    print(val)
    print("TEST")
    print(oot)

    Y_ = model2.predict(Xv)

    print("FIN _____________")

    filenames = [f"2_{company}_{filebase}best.jsonl", f"2_{company}best.jsonl"]
    save_validation_set(filenames, ohe, Xv, Yv, Y_)

    filenames = [f"2_{company}_{filebase}best.joblib", f"2_{company}_best.joblib"]
    for filename in filenames:
        joblib.dump(best_model, filename)


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
