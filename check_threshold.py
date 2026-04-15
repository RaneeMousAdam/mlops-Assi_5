import mlflow
import os
import sys

THRESHOLD = 0.85

def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Checking Run ID: {run_id}")

    # Fetch the run from MLflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: No accuracy metric found for this run.")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD}")

    if accuracy < THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")

if __name__ == "__main__":
    main()

