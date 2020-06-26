from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import os


if __name__ == "__main__":
    ## Search through runs based on experiment ID and pick top accuracy model run

    client = MlflowClient()

    run = client.search_runs(
        experiment_ids="1",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.acc_validation DESC"]
    )[0]

    # Get path to saved xgb model artifact
    # Could also here set tag for this model to "prod"
    xgb_model_path = f"runs:/{run.info.run_id}/model"

    #os.system(f"mlflow models serve -m {xgb_model_path} -p 5004")
    os.system(f"mlflow models build-docker -m {xgb_model_path} -n xgb_da_mote")
    os.system("docker run -p 5004:8080 xgb_da_mote")
    #os.system(f"mlflow sagemaker run-local -m {xgb_model_path} -i xgboost_mlflow_container -p 5004")
    #os.system(f"mlflow sagemaker deploy -a deployXgboostTest -m {xgb_model_path} -t ml.t2.medium -c 1 -i 998155714215.dkr.ecr.eu-central-1.amazonaws.com/benjaminscontainer:1.8.0 --region-name eu-central-1")