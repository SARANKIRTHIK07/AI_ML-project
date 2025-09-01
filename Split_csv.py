import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Input CSV path
input_path = r"C:\Users\user\Desktop\AI project\data\final_dataset_balanced.csv"

# Output folder
output_dir = r"C:\Users\user\Desktop\AI project\data\pythonProject\data"
os.makedirs(output_dir, exist_ok=True)

# Output file paths
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")

# Load dataset
df = pd.read_csv(input_path)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (80%) and test (20%) with stratification
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],  # Keeps category balance
    random_state=42
)

# Save datasets
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f" Train and test sets saved at:\n{train_path}\n{test_path}")
