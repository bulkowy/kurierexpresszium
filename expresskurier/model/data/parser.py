import json
import time
import copy
import random

from datetime import datetime, timedelta

with open("deliveries.jsonl","r") as d:
    deliveries = d.read().splitlines()

with open("sessions.jsonl","r") as s:
    sessions = s.read().splitlines()

with open("users.jsonl","r") as u:
    users = u.read().splitlines()

result = {
    360: {},
    516: {},
    620: {},
}
uid_city_map = {}

for user in users:
    user = json.loads(user)
    uid_city_map[user["user_id"]] = user["city"]
    #result[user["user_id"]] = {"city": user["city"], "deliveries": []}

prev_session = None
last_info = []

for session in sessions:
    session = json.loads(session)
    if len(last_info) > 0:
        if session["session_id"] == prev_session:
            if session["user_id"] is not None:
                for info in last_info:
                    if uid_city_map[session["user_id"]] in result[info[0]].keys():
                        result[info[0]][uid_city_map[session["user_id"]]].append(info[1])
                    else:
                        result[info[0]][uid_city_map[session["user_id"]]] = [info[1]]
                last_info.clear()
        else:
            last_info.clear()
    if session["purchase_id"] is not None:    
        time_sent = None
        time_delivery = None
        info = {}
        delivery_company = None
        for delivery in deliveries:
            delivery = json.loads(delivery)
            if delivery["purchase_id"] == session["purchase_id"]:
                if delivery["purchase_timestamp"] is not None:
                    time_sent = datetime.strptime(delivery["purchase_timestamp"],'%Y-%m-%dT%H:%M:%S')
                if delivery["delivery_timestamp"] is not None:
                    time_delivery = datetime.strptime(delivery["delivery_timestamp"],'%Y-%m-%dT%H:%M:%S.%f')
                if delivery["delivery_company"] is not None:
                    delivery_company = delivery["delivery_company"]
                break
        
        if time_sent is not None and time_delivery is not None and delivery_company is not None:
            delta_time = (time_delivery - time_sent) / timedelta(days=1)
            if delta_time > 0:
                info["delta_time"] = delta_time
                info["shipment_day"] = time_sent.weekday()
                info["hour"] = time_sent.hour
                if session["user_id"] is not None:
                    if uid_city_map[session["user_id"]] in result[delivery_company].keys():
                        result[delivery_company][uid_city_map[session["user_id"]]].append(info)
                    else:
                        result[delivery_company][uid_city_map[session["user_id"]]] = [info]
                else:
                    last_info.append((delivery_company, info))
            
    prev_session = session["session_id"]


random.seed(0)

for delivery_company in result.keys():
    training = []
    validation = []
    testing = []
    for city in result[delivery_company].keys():
        training_size = int(0.6 * len(result[delivery_company][city]))
        for i in range(training_size):
            index = random.randint(0,len(result[delivery_company][city])-1)
            delivery = result[delivery_company][city].pop(index)
            delivery["city"] = city
            training.append(delivery)
        validation_size = int(0.5 * len(result[delivery_company][city]))
        for i in range(validation_size):
            index = random.randint(0,len(result[delivery_company][city])-1)
            delivery = result[delivery_company][city].pop(index)
            delivery["city"] = city
            validation.append(delivery)
        for delivery in result[delivery_company][city]:
            delivery["city"] = city
            testing.append(delivery)

    filename = (f"{delivery_company}_training.jsonl")
    with open(filename, "w") as out:
        for delivery in training:
            out.write(json.dumps(delivery))
            out.write("\n")

    filename = (f"{delivery_company}_validation.jsonl")
    with open(filename, "w") as out:
        for delivery in validation:
            out.write(json.dumps(delivery))
            out.write("\n")
    
    filename = (f"{delivery_company}_testing.jsonl")
    with open(filename, "w") as out:
        for delivery in testing:
            out.write(json.dumps(delivery))
            out.write("\n")