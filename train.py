
# Data munging / maths
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# Plot stuff
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# Model
from xgboost import XGBClassifier

# Import OS stuff
import os

# Import mlflow stuff
import mlflow
import mlflow.xgboost

# Import packages
from utils.plot_learning import plot_learning
from utils.feature_engineering import feature_engineering

def log_xgboost(params, train_X, train_Y, test_X, test_Y):

    with mlflow.start_run() as ml_run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("state", "dev")
        xgc = XGBClassifier()
        xgc.set_params(**params)
        model = xgc.fit(train_X, train_Y, eval_set=[(train_X, train_Y), (test_X, test_Y)], eval_metric=['error', 'logloss'])
        predictions = model.predict(test_X)
        acc = accuracy_score(test_Y, predictions)
        loss = log_loss(test_Y, predictions)

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
        model.save_model("temp/model.pth")
        mlflow.log_artifact("temp/model.pth")
        return model, predictions, acc, loss



if __name__ == "__main__":

    # Read in the datas
    
    experiment_name = "churn_prediction"

    client = mlflow.tracking.MlflowClient()

    experiments = client.list_experiments()

    if experiment_name not in [experiment.name for experiment in experiments]:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    telcom_input = pd.read_csv(os.getcwd() + "/data/" + [l for l in os.listdir("data/") if l.endswith(".csv")][0])

    engineered = feature_engineering(telcom_input)
    telcom = engineered[0]
    Id_col = engineered[2]
    target_col = engineered[3]

    #splitting train and test data 
    train, test = train_test_split(telcom, test_size = .25 ,random_state = 111)
        
    ##seperating dependent and independent variables
    cols    = [i for i in telcom.columns if i not in Id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    test_X  = test[cols]
    test_Y  = test[target_col]

    #mlflow.xgboost.autolog()

    # Model

    maxDepthList = [7]
    gammaList = [0.1, 0.2]
    learningRateList = [0.001, 0.01, 0.1]
    nEstimatorsList = [50, 100, 150]

    for max_depth, gamma, learning_rate, n_estimators in [(max_depth, gamma, learning_rate, n_estimators) for max_depth in maxDepthList for gamma in gammaList for learning_rate in learningRateList for n_estimators in nEstimatorsList]:
        
        params = {
        # XGboost parameters
            'max_depth': max_depth,
            'gamma':gamma,
            'learning_rate': learning_rate,
            'colsample_bytree': 0.5,
            'n_estimators': n_estimators

        }

        model, predictions, accuracy, loss = log_xgboost(params, train_X, train_Y, test_X, test_Y)

        print(accuracy, loss)

