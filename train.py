import mlflow
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("digits-classifier")

    with mlflow.start_run() as run:
        # Load and split data
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        n_estimators = int(os.environ.get("N_ESTIMATORS", 1))
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")

        # Export run ID for the pipeline
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()
