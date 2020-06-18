
import pandas as pd
from utils.feature_engineering import feature_engineering
import requests
import s3fs
import boto3
import json

## Feature engineering "pipeline"
new_data = pd.read_csv("s3://bearingsight-ingest/new_data.csv")
new_data.drop('Unnamed: 0', inplace=True, axis=1)
engineered_new = feature_engineering(new_data)
telcom_new = engineered_new[0]
Id_col = engineered_new[2]
target_col = engineered_new[3] 
cols    = [i for i in telcom_new.columns if i not in Id_col + target_col]
inference_X  = telcom_new[cols]

print("K")

endpoint_name = "deployXgboostTest"
runtime = boto3.Session().client(service_name="runtime.sagemaker", region_name="eu-central-1")
payload = inference_X.iloc[0:2,:].to_json(orient="split")
response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json", Body=payload)
pred = json.loads(response["Body"].read().decode())

pd.DataFrame({"predictions": pred}).to_csv("s3://bearingsight-models/predictions.csv")
