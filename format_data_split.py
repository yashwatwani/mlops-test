import os
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = "data/raw_data.jsonl"
OUTPUT_DIR = "data/mlx_format"

# ⚠️ The instruction template required by the MLX script ⚠️
# We combine the three fields into one 'text' field.
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# --- Step 1: Format the Data ---
try:
    full_dataset = load_dataset("json", data_files=RAW_DATA_PATH, split="train")
except Exception as e:
    print(f"Error loading raw dataset: {e}")
    exit()

def format_data(example):
    """Combines instruction, input, and output into a single string under the 'text' key."""
    # This creates the single string the model expects to see
    formatted_text = ALPACA_PROMPT.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"]
    )
    return {"text": formatted_text}

# Apply the formatting and drop the old columns
formatted_dataset = full_dataset.map(format_data, remove_columns=["instruction", "input", "output"])
df = formatted_dataset.to_pandas()
print(f"Total formatted samples: {len(df)}")


# --- Step 2 & 3: Perform 80/10/10 Split (Same logic as before) ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# First Split: Test Set (10%)
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# Second Split: Train (80%) and Validation (10%)
train_df, validation_df = train_test_split(train_val_df, test_size=len(test_df), random_state=42, shuffle=True)

# Convert back and save
Dataset.from_pandas(train_df, preserve_index=False).to_json(os.path.join(OUTPUT_DIR, "train.jsonl"), orient="records", lines=True)
Dataset.from_pandas(validation_df, preserve_index=False).to_json(os.path.join(OUTPUT_DIR, "valid.jsonl"), orient="records", lines=True)
Dataset.from_pandas(test_df, preserve_index=False).to_json(os.path.join(OUTPUT_DIR, "test.jsonl"), orient="records", lines=True)

print("\n--- Final Files Saved ---")
print(f"✅ train.jsonl now contains the final 'text' field required by lora.py")