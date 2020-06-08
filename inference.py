
import pandas as pd
from utils.feature_engineering import feature_engineering
import requests

## Feature engineering pipeline
new_data = pd.read_csv("new_data/new_data.csv")
new_data.drop('Unnamed: 0', inplace=True, axis=1)
engineered_new = feature_engineering(new_data)
telcom_new = engineered_new[0]
Id_col = engineered_new[2]
target_col = engineered_new[3] 
cols    = [i for i in telcom_new.columns if i not in Id_col + target_col]
inference_X  = telcom_new[cols]

## Set up host
host = "0.0.0.0"
port = "1234"

url = f'http://{host}:{port}/invocations'

headers = {
    "Content-Type": "application/json",
}

http_data = inference_X.to_json(orient="split")

r = requests.post(url=url, headers=headers, data=http_data)

preds = []

for line in r.text.splitlines():
    preds.append(line)

preds = pd.DataFrame(preds)

preds.to_csv("results/predictions.csv")
print("Written to results/predictions.csv")