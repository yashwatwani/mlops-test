import subprocess
import mlflow
import os
import re # Required for parsing the loss string

# --- 0. Configuration and Credentials (KEEP THESE) ---
# NOTE: The token and username should ideally be set via OS environment variables 
# (export commands) rather than hardcoded here for security.
os.environ["MLFLOW_TRACKING_USERNAME"] = "yashwatwani28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d82940c2138e8370272889954f54b5647c92b9bf"

MLFLOW_URI = "https://dagshub.com/yashwatwani28/mlops-test.mlflow"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("politeness-bot-experiment")

# --- 1. Subprocess Streaming Function (NEW LOGIC) ---
def parse_and_log_stream(process, run_id):
    """Captures the output stream, parses for loss, and logs to MLflow."""
    # Regex to find the loss value in the MLX output line (e.g., "Loss: 0.123")
    LOSS_PATTERN = re.compile(r"Loss:\s*([\d\.]+)") 
    current_iter = 0

    print("--- Starting MLX Stream Parser ---")
    
    while True:
        # Read one line of standard output
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        
        if output:
            output = output.strip()
            print(output) # Print the line to the console for real-time viewing

            # Check for the loss value
            loss_match = LOSS_PATTERN.search(output)
            if loss_match:
                try:
                    loss_value = float(loss_match.group(1))
                    current_iter += 1 
                    
                    # LOG THE METRIC TO DAGSHUB
                    mlflow.log_metric("train_loss", loss_value, step=current_iter)
                    
                except:
                    pass

# --- 2. Define Command and Paths (MODIFIED) ---
MODEL_PATH = "/Users/yash.watwani/Documents/mlops-test/mlx-models/tinyllama"
ADAPTERS_PATH = "/Users/yash.watwani/Documents/mlops-test/adapters"
ITERATIONS = "10" 
DATA_PATH = "/Users/yash.watwani/Documents/mlops-test/data/mlx_format"

# The command to execute the MLX training (we will pass this to Popen)
MLX_COMMAND = [
    "python", 
    "mlx-examples/lora/lora.py", 
    "--model", MODEL_PATH,
    "--train",
    "--data", DATA_PATH,
    "--adapter-file", f"{ADAPTERS_PATH}/politeness_adapters.npz",
    "--iters", ITERATIONS,
    "--batch-size", "1",
]


# --- 3. Execute Training within MLflow Run (NEW EXECUTION BLOCK) ---
with mlflow.start_run() as run:
    print(f"Starting MLflow Run ID: {run.info.run_id}")
    
    # Log key parameters explicitly
    mlflow.log_param("model_type", "MLX_LoRA_TinyLlama")
    mlflow.log_param("iterations", ITERATIONS)

    # 1. Start the MLX script using Popen (for streaming)
    try:
        process = subprocess.Popen(
            MLX_COMMAND, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            cwd=os.getcwd()
        )

        # 2. Run the parser to capture and log metrics
        parse_and_log_stream(process, run.info.run_id)
        
        # 3. Wait for the subprocess to finish and check status
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "MLX Training Failed")
            
        # 4. Log final artifacts
        mlflow.log_artifact(f"{ADAPTERS_PATH}/politeness_adapters.npz", artifact_path="model_adapters")
        print(f"✅ Run successful. Artifacts logged.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")