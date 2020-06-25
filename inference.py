
import pandas as pd
from utils.load_data import load_data
import requests
import json

def get_predictions(host, port, data):
    """ Takes a host ip, port and pandas dataframe and returns predictions from model endpoint."""
    # Serving URL
    url = f'http://{host}:{port}/invocations'
    
    # Header, format defaults to pandas-split
    headers = {
    'Content-Type': 'application/json'
    }

    # Transform df to split oriented json
    http_data = data.to_json(orient='split')

    # Send post request
    r = requests.post(url=url, headers=headers, data=http_data)
    print(r)

    # Return predictions
    return pd.DataFrame({"predictions": json.loads(r.text)})


if __name__ == "__main__":
    ## Feature engineering "pipeline"
    inference_X = load_data("new_data/new_data.csv", inference = True)[0]

    # Set host and port
    host = "0.0.0.0"
    port = "5004"

    # Get predictions
    predictions = get_predictions(host, port, inference_X)
    print(predictions.head())

    # Write predictions to csv
    predictions.to_csv("predictions/predictions.csv")
    print("Wrote predictions to predicitons/predictions.csv")
