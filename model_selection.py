
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import os

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    client = mlflow.tracking.client.MlflowClient()

    run = client.search_runs(
        experiment_ids="1",
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.accuracy DESC"]
    )[0]

    # Get path to saved xgb model artifact
    # Could also here set tag for this model to "prod"
    xgb_model_path = "runs:/{}/model".format(run.info.run_id)

    print(os.system("export MODEL_PATH={}".format(xgb_model_path)))
