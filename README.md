# Telco churn ML Lifecycle Management example using MLflow
This repo contains an example file structure and workflow for ML Lifecycle Management using MLflow.
The example uses the "Telco Churn" dataset from Kaggle (https://www.kaggle.com/blastchar/telco-customer-churn).
See https://mlflow.org/docs/latest/index.html for MLflow documentation.

## Workflow
The contents of this repo will:
- Perform simple grid search to find "optimal" xgboost model
- Log parameters, metrics and artifacts (including model) to local MLflow directory
- Pick the best model and "package" it for use with MLflow deployment options
- Perform inference on "new data" using POST-request to model serving Rest API (using "mlflow models serve")

MLflow filestructure saved locally. Run mlflow ui from within repo to open the MLflow UI and see experiment logs.

Model training (simple grid search xgboost): train.py
Model selection and "packaging": model_selection.py
Model inference (given "mlflow model serve" is used): inference.py

## How to
1. ```git clone https://github.com/benaan123/mlflow_telco_churn.git```
2. ```conda create -n "my_env" python=3.7```
4. ```conda activate "my_env"```
3. ```pip install -r requirements.txt```
4. ```python train.py``` or run through train_notebook.ipynb to fill mlruns folder with runs
5. ```mlflow ui``` to run mlflow ui on localhost. Alternatively ```mlflow server --h 0.0.0.0 -p 1234``` to run mlflow server locally on port 1234.
6. ```python model_selection.py``` to find best model run path
7. ```mlflow models serve -m {run_id}/model -p 5004``` to serve model on localhost:5004. Alterenatively, get the run id from the best model and run ```mlflow models build-docker -m {run_id}/model -n "my_model_name"``` followed by ```docker run "my_model_name" -p 5004:8000``` to serve model through a docker container on localhost:5004. ```mlflow sagemaker run-local -m {run_id}/model -p 5004``` to run using sagemaker-compatible docker container on port localhost:5004. See ```deployment_commands.txt``` for how to deploy to sagemaker endpoint.
8. ```python inference.py``` to perform inference on test data and save in predictions/ folder. Change if sagemaker endpoint is used.

