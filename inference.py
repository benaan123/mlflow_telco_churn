
import pandas as pd
from utils.feature_engineering import feature_engineering
import requests
import json

## Feature engineering "pipeline"
new_data = pd.read_csv("new_data/new_data.csv")
new_data.drop('Unnamed: 0', inplace=True, axis=1)
engineered_new = feature_engineering(new_data)
telcom_new = engineered_new[0]
Id_col = engineered_new[2]
target_col = engineered_new[3] 
cols    = [i for i in telcom_new.columns if i not in Id_col + target_col]
inference_X  = telcom_new[cols]

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
    print(r.text)
    return pd.DataFrame({"predictions": json.loads(r.text)})

predictions = get_predictions(host, port, inference_X)

print(predictions.head())

predictions.to_csv("predictions/predictions.csv")
print("Wrote predictions to predicitons/predictions.csv")
