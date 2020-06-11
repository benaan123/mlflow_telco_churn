import pickle
import xgboost as xgb
import mlflow.pyfunc
from mlflow.xgboost import load_model
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import sys

from utils.feature_engineering import feature_engineering

## Search through runs based on experiment ID and pick top accuracy model run
run = MlflowClient().search_runs(
    experiment_ids="5",
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.accuracy DESC"]
)[0]

# Get path to saved xgb model artifact
# Could also here set tag for this model to "prod"
xgb_model_path = f"runs:/{run.info.run_id}/model"


class XGBWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = load_model(xgb_model_path)
    
    def predict(self, context, model_input):
        """Predict method that also does feature engineering"""
        return self.xgb_model.predict(model_input)

# Create condaenv
import cloudpickle
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python=3.7.6",
        "cloudpickle={}".format(cloudpickle.__version__)
    ],
    "name": "xgb_env"
}

# Save the mlflow model, this can be directly used to serve predictions using mlflow CLI
mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path, python_model=XGBWrapper(),
    conda_env=conda_env
)


