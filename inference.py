
import pandas as pd
from utils.load_data import load_data
import requests
import json

## Feature engineering "pipeline"
inference_X = load_data("new_data/new_data.csv", inference = True)[0]

# Set host and port
#host = "127.0.0.1"
#port = "5001"
host = "0.0.0.0"
port = "5004"

def get_predictions(host, port, data):
    """ Takes a host ip, port and pandas dataframe and returns predictions from model endpoint."""
    url = f'http://{host}:{port}/invocations'
    headers = {
    'Content-Type': 'application/json'
    }
    http_data = data.to_json(orient='split')
    r = requests.post(url=url, headers=headers, data=http_data)
    print(r)
    return pd.DataFrame({"predictions": json.loads(r.text)})

predictions = get_predictions(host, port, inference_X)

print(predictions.head())

predictions.to_csv("predictions/predictions.csv")
print("Wrote predictions to predicitons/predictions.csv")
