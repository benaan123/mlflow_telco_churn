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
import matplotlib as plt

# Import OS stuff
import os

# Import mlflow stuff
import mlflow
import mlflow.xgboost

# Import utils
from utils.plot_learning import plot_learning
from utils.feature_engineering import feature_engineering

def log_xgboost(params, train_X, train_Y, validation_X, validation_Y):
    """ Takes a set of xgboost parameters, trains a model and logs parameters, metrics, artifacts and the model itself.

        args:
            params: Dictionary of xgboost parameters
            train_X: pandas dataframe containing training features
            train_Y: pandas dataframe containing training labels
            test_X: pandas data
    """
    with mlflow.start_run() as ml_run:
        
        # Her logger vi alle parameterene paa vei inn
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Setter tag, her kan beste modell settes til "prod" senere
        mlflow.set_tag("state", "dev")
        
        xgc = XGBClassifier()
        xgc.set_params(**params)
        model = xgc.fit(train_X, train_Y.values.ravel(), eval_set=[(train_X, train_Y.values.ravel()), (validation_X, validation_Y.values.ravel())], eval_metric=['error', 'logloss'], verbose=0)
        
        predictions = model.predict(validation_X)
        acc = accuracy_score(validation_Y.values.ravel(), predictions)
        loss = log_loss(validation_Y.values.ravel(), predictions)

        ## Logging av ulike plots

        # Trening/valideringserror
        error_plot = plot_learning(model, "error")
        error_plot.savefig("temp/error_plot.png")
        mlflow.log_artifact("temp/error_plot.png")
        # Trening/valideringsloss
        loss_plot = plot_learning(model, "logloss")
        loss_plot.savefig("temp/logloss.png")
        mlflow.log_artifact("temp/logloss.png")
        # Confusion matrix for klassifisering
        conf_mat = confusion_matrix(validation_Y, predictions)
        conf_mat_plot = sns.heatmap(conf_mat, annot=True, fmt='g')
        conf_mat_plot.figure.savefig("temp/confmat.png")
        mlflow.log_artifact("temp/confmat.png")

        # Logging av loss og accuracy verdier
        mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})
        
        # Logging av selve modellen
        mlflow.xgboost.log_model(model, "model")
        
        print(f"Model trained with parameters: {params}")
        
        return model, predictions, acc, loss



if __name__ == "__main__":

    plt.rcParams.update({'figure.max_open_warning': 0})

    experiment_name = "da_demo_xgboost"

    #tracking_uri = "http://0.0.0.0:5000"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment(experiment_name)

    experiments = client.list_experiments()

    if experiment_name not in [experiment.name for experiment in experiments]:
        mlflow.create_experiment(experiment_name)

    telcom_input = pd.read_csv("data/telco_churn.csv")

    engineered = feature_engineering(telcom_input)
    telcom = engineered[0]
    Id_col = engineered[2]
    target_col = engineered[3]

    #splitting train and test data 
    train, validation = train_test_split(telcom, test_size = .25 ,random_state = 111)
        
    ##seperating dependent and independent variables
    cols    = [i for i in telcom.columns if i not in Id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    validation_X  = validation[cols]
    validation_Y  = validation[target_col]
    
    # Grid search
    max_depth_list = [7]
    gamma_list = [0.1, 0.2]
    learning_rate_list = [0.001, 0.01, 0.1]
    n_estimators_list = [150, 200, 250]

    for max_depth in max_depth_list:
        for gamma in gamma_list:
            for learning_rate in learning_rate_list:
                for n_estimators in n_estimators_list:
        
                    params = {
                        'max_depth': max_depth,
                        'gamma':gamma,
                        'learning_rate': learning_rate,
                        'colsample_bytree': 0.5,
                        'n_estimators': n_estimators,
                        'n_jobs': -1

                    }

                    model, predictions, accuracy, loss = log_xgboost(params, train_X, train_Y, validation_X, validation_Y)

                    print(accuracy, loss)

