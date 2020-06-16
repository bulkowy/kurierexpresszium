import json
import time
import copy
from datetime import datetime, timedelta

with open("deliveries.jsonl","r") as d:
    deliveries = d.read().splitlines()

with open("sessions.jsonl","r") as s:
    sessions = s.read().splitlines()

with open("users.jsonl","r") as u:
    users = u.read().splitlines()

result = {}

for user in users:
    user = json.loads(user)
    result[user["user_id"]] = {"city": user["city"], "deliveries": []}

prev_session = None
last_info = []

for session in sessions:
    session = json.loads(session)
    if len(last_info) > 0:
        print(last_info)
        if session["session_id"] == prev_session:
            if session["user_id"] is not None:
                print(f"u: {session['user_id']} s: {session['session_id']}")
                for info in last_info:
                    result[session["user_id"]]["deliveries"].append(info)
                last_info.clear()
        else:
            last_info.clear()
    if session["purchase_id"] is not None:    
        time_sent = None
        time_delivery = None
        info = {}
        for delivery in deliveries:
            delivery = json.loads(delivery)
            if delivery["purchase_id"] == session["purchase_id"]:
                if delivery["purchase_timestamp"] is not None:
                    time_sent = datetime.strptime(delivery["purchase_timestamp"],'%Y-%m-%dT%H:%M:%S')
                if delivery["delivery_timestamp"] is not None:
                    time_delivery = datetime.strptime(delivery["delivery_timestamp"],'%Y-%m-%dT%H:%M:%S.%f')
                if delivery["delivery_company"] is not None:
                    info = {"delivery_company": delivery["delivery_company"]}
                break
        
        if time_sent is not None and time_delivery is not None and "delivery_company" in info.keys():
            delta_time = (time_delivery - time_sent) / timedelta(days=1)
            if delta_time > 0:
                info["delta_time"] = delta_time
                info["shipment_day"] = time_sent.weekday()
                if session["user_id"] is not None:
                    result[session["user_id"]]["deliveries"].append(info)
                else:
                    last_info.append(info)
            
    prev_session = session["session_id"]

full_result = {}
for res in result.keys():
    if result[res]["city"] in full_result.keys():
        full_result[result[res]["city"]] += result[res]["deliveries"]
    else:
        full_result[result[res]["city"]] = copy.deepcopy(result[res]["deliveries"])


with open("data.jsonl", "w") as out:
    for city in full_result.keys():
        out.write("{\"")
        out.write(f"{city}\": ")
        out.write(json.dumps(full_result[city]))
        out.write("}\n")
