import argparse
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

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

def encode_city(city: str):
    mapping ={
        "Gdynia": np.array([0,0,0]),
        "Konin": np.array([0,0,1]),
        "Kutno": np.array([0,1,0]),
        "Mielec": np.array([0,1,1]),
        "Police": np.array([1,0,0]),
        "Radom": np.array([1,0,1]),
        "Szczecin": np.array([1,1,0]),
        "Warszawa": np.array([1,1,1]),
    }

    return mapping[city]

def decode_city(city):
    if city[0] == 0:
        if city[1] == 0:
            if city[2] == 0:
                return "Gdynia"
            else:
                return "Konin"
        else:
            if city[2] == 0:
                return "Kutno"
            else:
                return "Mielec"
    else:
        if city[1] == 0:
            if city[2] == 0:
                return "Police"
            else:
                return "Radom"
        else:
            if city[2] == 0:
                return "Szczecin"
            else:
                return "Warszawa"


def training(company: int):
    Xt = []
    Yt = []
    Xv = []
    Yv = []
    Xtst = []
    Ytst = []
    filename = f"data/{company}_training.jsonl"
    with open(filename, "r") as t:
        training_data = t.read().splitlines()

    for line in training_data:
        data = json.loads(line)
        Xt.append(encode_city(data["city"]))
        Yt.append(np.float32(data["delta_time"]))
    
    filename = f"data/{company}_validation.jsonl"
    with open(filename, "r") as v:
        validation_data = v.read().splitlines()

    for line in validation_data:
        data = json.loads(line)
        Xv.append(encode_city(data["city"]))
        Yv.append(np.float32(data["delta_time"]))
        
    
    filename = f"data/{company}_testing.jsonl"
    with open(filename, "r") as v:
        testing_data = v.read().splitlines()

    for line in testing_data:
        data = json.loads(line)
        Xtst.append(encode_city(data["city"]))
        Ytst.append(np.float32(data["delta_time"]))


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

    clf = GridSearchCV(model, hyperparameters, cv=10)
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

    print(test)
    print(val)
    print(oot)

    Y_ = model2.predict(Xv)

    print("FIN _____________")

    filename = f"{company}_{filebase}best.jsonl"
    with open(filename, "w+") as result:
        for xv, yv, y_ in zip(Xv,Yv,Y_):
            result.write(json.dumps({"city": decode_city(xv), "delivery_time": float(yv), "predicted_time": float(y_)}))
            result.write("\n")

    filename = f"{company}_{filebase}best.joblib"
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
