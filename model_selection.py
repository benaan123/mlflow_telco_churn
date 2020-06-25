import xgboost as xgb
import mlflow.pyfunc
from mlflow.xgboost import load_model
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import datetime
import os

from utils.feature_engineering import feature_engineering

## Search through runs based on experiment ID and pick top accuracy model run

if __name__ == "__main__":
    client = MlflowClient()

    run = client.search_runs(
        experiment_ids="2",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.accuracy DESC"]
    )[0]

    # Get path to saved xgb model artifact
    # Could also here set tag for this model to "prod"
    xgb_model_path = f"runs:/{run.info.run_id}/model"

    os.system(f"mlflow models serve -m {xgb_model_path} -p 5004")
