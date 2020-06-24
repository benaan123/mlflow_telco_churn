
# Data munging / maths
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# Model
from xgboost import XGBClassifier

# Import seaborn
import seaborn as sns

# Import OS stuff
import os

# Import mlflow stuff
import mlflow
import mlflow.xgboost

# S3
import s3fs

# Import packages
from utils.plot_learning import plot_learning
from utils.load_data import load_data

def log_xgboost(params, train_X, train_Y, test_X, test_Y):

    with mlflow.start_run() as ml_run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("state", "dev")
        xgc = XGBClassifier(objective="binary:logistic")
        xgc.set_params(**params)
        model = xgc.fit(train_X, train_Y.values.ravel(), eval_set=[(train_X, train_Y.values.ravel()), (test_X, test_Y.values.ravel())], eval_metric=['error', 'logloss'], verbose=0)
        predictions = model.predict(test_X)
        acc = accuracy_score(test_Y.values.ravel(), predictions)
        loss = log_loss(test_Y.values.ravel(), predictions)

        ## Plots
        error_plot = plot_learning(model, "error")
        error_plot.savefig("temp/error_plot.png")
        mlflow.log_artifact("temp/error_plot.png")
        loss_plot = plot_learning(model, "logloss")
        loss_plot.savefig("temp/logloss.png")
        mlflow.log_artifact("temp/logloss.png")
        conf_mat = confusion_matrix(test_Y, predictions)
        conf_mat_plot = sns.heatmap(conf_mat, annot=True, fmt='g')
        conf_mat_plot.figure.savefig("temp/confmat.png")
        mlflow.log_artifact("temp/confmat.png")
        mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})
        
        mlflow.xgboost.log_model(model, "model")
        
        print(f"Model trained with parameters: {params}")
        
        return model, predictions, acc, loss



if __name__ == "__main__":

    # Read in the datas
    train_X, train_Y, test_X, test_Y = load_data("s3://bearingsight-ingest/telco_churn.csv", test_size = 0.25)

    experiment_name = "s3_mlflow"

    tracking_uri = "http://0.0.0.0:5000"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment(experiment_name)

    experiments = client.list_experiments()

    if experiment_name not in [experiment.name for experiment in experiments]:
        mlflow.create_experiment(experiment_name)
                
    max_depth_list = [5]
    colsample_bytree_list = [0.3, 0.5, 0.8]
    learning_rate_list = [0.1, 0.2]
    n_estimators_list = [200, 250, 300]

    for max_depth in max_depth_list:
        for colsample_bytree in colsample_bytree_list:
            for learning_rate in learning_rate_list:
                for n_estimators in n_estimators_list:
                    params = {
                    # XGboost parameters
                        'max_depth': max_depth,
                        'gamma': 0,
                        'learning_rate': learning_rate,
                        'colsample_bytree': colsample_bytree,
                        'n_estimators': n_estimators,
                        'n_threads': -1

                    }

                    model, predictions, accuracy, loss = log_xgboost(params, train_X, train_Y, test_X, test_Y)

                    print(accuracy, loss)

