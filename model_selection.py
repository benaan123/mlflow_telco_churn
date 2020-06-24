import xgboost as xgb
import mlflow.pyfunc
from mlflow.xgboost import load_model
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import datetime

from utils.feature_engineering import feature_engineering

## Search through runs based on experiment ID and pick top accuracy model run
client = MlflowClient()

#mlflow.set_tracking_uri("http://0.0.0.0:5000")

run = client.search_runs(
    experiment_ids="1",
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
        self.xgb_model = load_model(xgb_model_path)
    
    def predict(self, context, model_input):
        """Predict method that also does feature engineering"""
        model_input_dmatrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(model_input_dmatrix)


# Create condaenv
import cloudpickle
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python=3.7.6",
        "cloudpickle={}".format(cloudpickle.__version__),
        {"pip": ["xgboost==1.1.1", "mlflow"]}
    ],
    "name": "xgb_env"
}

# Save the mlflow model, this can be directly used to serve predictions using mlflow CLI
mlflow_pyfunc_model_path = "xgboost_"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path, python_model=XGBWrapper(),
    conda_env=conda_env
)


