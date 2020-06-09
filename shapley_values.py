import pickle
import xgboost as xgb
import shap
import pandas as pd
import datetime

from utils.feature_engineering import feature_engineering

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


## Search through runs based on experiment ID and pick top accuracy model run
run = MlflowClient().search_runs(
    experiment_ids="0",
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.accuracy DESC"]
)[0]

# Get path to saved xgb model artifact
# Could also here set tag for this model to "prod"
xgb_model_path = run.info.artifact_uri + "/model.pth"

artifacts = {
    "xgb_model": xgb_model_path
}

# Get path to saved xgb model artifact
# Could also here set tag for this model to "prod"
model = xgb.XGBClassifier()
model.load_model(artifacts["xgb_model"])

new_data = pd.read_csv("new_data/new_data.csv")
new_data.drop('Unnamed: 0', inplace=True, axis=1)

engineered_new = feature_engineering(new_data)
telcom_new = engineered_new[0]
Id_col = engineered_new[2]
target_col = engineered_new[3] 

cols    = [i for i in telcom_new.columns if i not in Id_col + target_col]

inference_X  = telcom_new[cols]

inference_X['predictions'] = model.predict(inference_X)
inference_X.to_csv(f"results/model_results_")

shap_X = inference_X.sample(50)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_X)

shap.decision_plot(explainer.expected_value, shap_values, shap_X, ignore_warnings=True, link="logit")

# Example of single case
shap.decision_plot(explainer.expected_value, shap_values[1], shap_X.iloc[1], link='logit', highlight=0)
