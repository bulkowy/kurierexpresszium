import argparse
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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


def training(company: int):
    #features = ['shipment_day', 'city']
    features = ['city']
    target = ['delta_time']
    ohe = OneHotEncoder(sparse=False)

    filename = f"data/{company}_training.jsonl"
    Xt, Yt = load_data(filename, features, target)
    ohe.fit(Xt)
    Xt = ohe.transform(Xt)
    joblib.dump(ohe, 'ohe.joblib')
    
    filename = f"data/{company}_validation.jsonl"
    Xv, Yv = load_data(filename, features, target)
    Xv = ohe.transform(Xv)
        
    
    filename = f"data/{company}_testingA.jsonl"
    XA, YA = load_data(filename, features, target)

    filename = f"data/{company}_testingB.jsonl"
    XB, YB = load_data(filename, features, target)

    Xtst = XA + XB
    Ytst = YA + YB
    
    Xtst = ohe.transform(Xtst)


    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 30))
    p = [1, 2]
    algorithm = ['ball_tree', 'kd_tree']
    weights = ['uniform', 'distance']

    hyperparameters = dict(
        leaf_size=leaf_size, 
        n_neighbors=n_neighbors, 
        p=p,
        algorithm=algorithm,
        weights=weights,
    )
    model = KNeighborsRegressor()
    model.fit(Xt, Yt)

    test = create_measures(Yt,model.predict(Xt))
    val = create_measures(Yv,model.predict(Xv))
    oot = create_measures(Ytst,model.predict(Xtst)) 

    print("PRE GRIDSEARCH _____________")

    print("TEST")
    print(test)
    print("VAL")
    print(val)
    print("TEST")
    print(oot)

    clf = GridSearchCV(model, hyperparameters, cv=10, n_jobs=4)
    best_model = clf.fit(Xv,Yv)

    print("POST GRIDSEARCH _____________")

    params = best_model.best_estimator_.get_params()

    print('Best leaf_size:', params['leaf_size'])
    print('Best p:', params['p'])
    print('Best n_neighbors:', params['n_neighbors'])
    print('Best algorithm:', params['algorithm'])
    print('Best weights:', params['weights'])

    filebase = ""
    for k, v in params.items():
        filebase += f"{v}_"
    
    model2 = KNeighborsRegressor(**params)
    model2.fit(Xt, Yt)

    test = create_measures(Yt,model2.predict(Xt))
    val = create_measures(Yv,model2.predict(Xv))
    oot = create_measures(Ytst,model2.predict(Xtst)) 

    print("TEST")
    print(test)
    print("VAL")
    print(val)
    print("TEST")
    print(oot)

    Y_ = model2.predict(Xv)

    print("FIN _____________")

    filenames = [f"{company}_{filebase}best.jsonl", f"{company}best.jsonl"]
    save_validation_set(filenames, ohe, Xv, Yv, Y_)

    filenames = [f"{company}_{filebase}best.joblib", f"{company}_best.joblib"]
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
