import mlflow
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

# --- 1. MLOps Configuration ---
# NOTE: Ensure these environment variables are exported in your terminal BEFORE running the script
# export MLFLOW_TRACKING_USERNAME="yashwatwani28"
# export MLFLOW_TRACKING_PASSWORD="YOUR_DAGSHUB_TOKEN"

MLFLOW_URI = "https://dagshub.com/yashwatwani28/mlops-test.mlflow"
EXPERIMENT_NAME = "Sklearn_Model_Comparison_Iris_Tough_Test" # Renamed experiment
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2. Data Preparation ---
iris = load_iris()
X, y = iris.data, iris.target

# --- 3. Model Training Function ---
def train_and_log_model(model_name, model_instance, params, run_seed):
    """Trains a model, calculates metrics, and logs everything to MLflow."""
    
    # NEW: Split data inside the loop with a specific seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.6, # Drastically reduce training data to make it harder
        random_state=run_seed,
        shuffle=True
    )
    
    # Start a nested MLflow run for each individual model training
    # FIX: Added nested=True to link this run to the current active parent run.
    with mlflow.start_run(run_name=f"{model_name}_Seed_{run_seed}", nested=True) as run:
        print(f"--- Starting Run for: {model_name} (Run ID: {run.info.run_id}) ---")
        
        # Log the critical run seed
        mlflow.log_param("data_split_seed", run_seed)
        mlflow.log_param("train_set_size", len(X_train))
        
        # Train the model
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # --- MLflow Logging ---
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        signature = infer_signature(X_train, y_pred)
        
        mlflow.sklearn.log_model(
            sk_model=model_instance,
            artifact_path="model",
            signature=signature,
            registered_model_name="IrisClassifierRegistry" # Register model for comparison
        )
        
        print(f"Metrics logged: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        return accuracy

# --- 4. Define Hyperparameter Runs ---
# NOTE: We keep these generic to be instantiated inside the loop with the new seed

base_model_runs = [
    # RUN 1: Logistic Regression (Basic)
    {
        "name": "Logistic_Regression_Basic",
        "class": LogisticRegression,
        "params": {"max_iter": 100, "solver": "lbfgs", "penalty": "l2"},
    },
    # RUN 2: Random Forest (Shallow)
    {
        "name": "Random_Forest_Shallow",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 50, "max_depth": 3, "criterion": "gini"},
    },
    # RUN 3: Random Forest (Deep)
    {
        "name": "Random_Forest_Deep",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": "None", "criterion": "gini"},
    },
    # RUN 4: Support Vector Machine (C-tuned)
    {
        "name": "SVM_C_Tuned",
        "class": SVC,
        "params": {"kernel": "linear", "C": 0.5},
    },
]

# --- 5. Execute All Runs and Find Best Model ---
# We will run the sweep over different data splits (seeds)
SWEEP_SEEDS = [0, 42, 88] # Running the sweep three times

best_global_accuracy = 0
best_global_model = None

# Wrap all runs in a parent MLflow run for organizational purposes
with mlflow.start_run(run_name="Hyperparameter_Sweep_Parent_Run"):
    for run_seed in SWEEP_SEEDS:
        for run_config in base_model_runs:
            
            # Instantiate the model inside the loop to reset random states
            model_instance = run_config["class"](random_state=42, **{k: v for k, v in run_config["params"].items() if k != 'max_depth' or v != 'None'})
            
            # Special handling for max_depth='None'
            if run_config["name"] == "Random_Forest_Deep":
                model_instance.set_params(max_depth=None)

            current_accuracy = train_and_log_model(
                run_config["name"],
                model_instance,
                run_config["params"],
                run_seed # Pass the current seed used for the data split
            )
            
            # Check for the best performing model overall
            if current_accuracy > best_global_accuracy:
                best_global_accuracy = current_accuracy
                best_global_model = run_config["name"]

    print("\n=============================================")
    print(f"MLflow Tough Sweep Complete.")
    print(f"Best Global Model (Highest Accuracy in any seed): {best_global_model}")
    print(f"Find all 12 runs and compare charts on Dagshub.")
    print("=============================================")

# --- 6. Promote Best Model to Production (Demonstrating Model Registry) ---
# NOTE: This section is commented out but shows the necessary step for production governance.

# client = MlflowClient()
# # Logic to find the best run ID from the 12 total runs (based on accuracy)
# best_run = client.search_runs(
#     experiment_ids=["1"], # Assuming experiment ID 1
#     order_by=["metrics.accuracy DESC"], 
#     max_results=1
# )[0]

# latest_model_version = client.search_model_versions(f"name='IrisClassifierRegistry'")[0].version
# client.transition_model_version_stage(
#     name="IrisClassifierRegistry",
#     version=latest_model_version,
#     stage="Production"
# )