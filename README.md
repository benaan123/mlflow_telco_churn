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

## Todo
- Finish shapley value script to get shap values from a given model run.
- Potentially add shapley decision plot or similar to rest API.

