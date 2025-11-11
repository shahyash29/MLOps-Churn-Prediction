# tools/promote_best.py  (or training/promote_best.py if you prefer)
import os
from dotenv import load_dotenv
load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_exp")
MODEL_NAME          = os.getenv("MODEL_NAME", "churn")
TARGET_STAGE        = os.getenv("TARGET_STAGE", "Staging")  # Staging or Production

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def archive_existing_in_stage(client: MlflowClient, model_name: str, stage: str):
    """Archive all existing versions currently in `stage`."""
    for mv in client.get_latest_versions(model_name, [stage]):
        if mv.current_stage == stage:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )
            print(f"Archived {model_name} v{mv.version} from {stage}")

def main():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp, f"Experiment `{EXPERIMENT_NAME}` not found."

    # pick best run by AUC
    best = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="metrics.roc_auc > 0",
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    ).iloc[0]
    run_id = best.run_id
    print(f"Best run: {run_id}  roc_auc={best['metrics.roc_auc']:.4f}")

    client = MlflowClient()

    # register from the run's artifact path (requires that you logged `artifact_path="model"`)
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    print(f"Registered {MODEL_NAME} version: {mv.version}")

    # optional: ensure only one version stays in the target stage
    try:
        archive_existing_in_stage(client, MODEL_NAME, TARGET_STAGE)
    except Exception as e:
        print(f"(Skipping manual archive step) {e!r}")

    # promote new version to target stage (compat: no archive_existing kw)
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage=TARGET_STAGE
    )
    print(f"Promoted {MODEL_NAME} v{mv.version} → {TARGET_STAGE}")

if __name__ == "__main__":
    main()
