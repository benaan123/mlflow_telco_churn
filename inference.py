
import pandas as pd
from utils.load_data import load_data
import requests
import s3fs
import boto3
import json

## Feature engineering "pipeline"
inference_X = load_data("s3://bearingsight-ingest/new_data.csv", inference=True)[0]

# Set host and port
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

predictions.to_csv("s3://bearingsight-models/predictions.csv")
print("Wrote predictions to s3://bearingsight-models/predictions.csv")