import argparse
import json
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import joblib

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

def decode_city(city: str):
    mapping ={
        np.array([0,0,0]): "Gdynia",
        np.array([0,0,1]): "Konin",
        np.array([0,1,0]): "Kutno",
        np.array([0,1,1]): "Mielec",
        np.array([1,0,0]): "Police",
        np.array([1,0,1]): "Radom",
        np.array([1,1,0]): "Szczecin",
        np.array([1,1,1]): "Warszawa",
    }

    return mapping[city]

def training(company: int, n_neighbors: int, weights: str):
    Xt = []
    Yt = []
    Xv = []
    Yv = []
    filename = f"data/{company}_training.jsonl"
    with open(filename, "r") as t:
        training_data = t.read().splitlines()

    for line in training_data:
        data = json.loads(line)
        Xt.append(encode_city(data["city"]))
        Yt.append(np.float32(data["delivery_time"]))
    
    filename = f"data/{company}_validation.jsonl"
    with open(filename, "r") as v:
        validation_data = v.read().splitlines()

    for line in validation_data:
        data = json.loads(line)
        Xv.append(encode_city(data["city"]))
        Yv.append(np.float32(data["delivery_time"]))

    knn = KNeighborsRegressor(n_neighbors, weights=weights)
    knn.fit(Xt,Yt)

    Y_ = knn.predict(Xv)
    score = knn.score(Xv, Yv)

    filename = f"experiments/{company}_{n_neighbors}_{weights}_{score}.jsonl"
    with open(filename, "w") as result:
        for xv, yv, y_ in zip(Xv,Yv,Y_):
            result.write(json.dumps({"city": decode_city(xv), "delivery_time": yv, "predicted_time": y_}))
            result.write("\n")

    filename = f"experiments/{company}_{n_neighbors}_{weights}.joblib"
    joblib.dump(knn, filename)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-c",
        "--company",
        action = "store",
        choices = [360, 516, 620],
        required = True        
    )
    argument_parser.add_argument(
        "-n",
        "--neighbors",
        action = "store",
        type = int,
        required = True        
    )
    argument_parser.add_argument(
        "-w",
        "--weights",
        action = "store",
        choices = ['uniform', 'distance'],
        required = True        
    )
    program_args = vars(argument_parser.parse_args())
    
    training(program_args["company"], program_args["neighors"], program_args["weights"])