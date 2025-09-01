import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Paths
# ----------------------------
model_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\models\classifier_model"
test_data_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\data\test.csv"
output_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\Output"
os.makedirs(output_path, exist_ok=True)

# ----------------------------
# Load model, tokenizer, label encoder
# ----------------------------
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Load test dataset
# ----------------------------
test_df = pd.read_csv(test_data_path)
texts = test_df["text"].tolist()
labels = label_encoder.transform(test_df["label"].tolist())  # encode labels

# ----------------------------
# Dataset & DataLoader
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} | {"labels": self.labels[idx]}

test_dataset = TextDataset(texts, labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------------------------
# Run predictions
# ----------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ----------------------------
# Overall metrics
# ----------------------------
overall_accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
average_confidence = np.mean(np.max(all_probs, axis=1))
num_test_cases = len(all_labels)

metrics = {
    "accuracy": overall_accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "average_confidence": average_confidence,
    "num_test_cases": num_test_cases
}
pd.DataFrame([metrics]).to_csv(os.path.join(output_path, "overall_metrics.csv"), index=False)
print("Overall Metrics:", metrics)

# ----------------------------
# Category-wise accuracy
# ----------------------------
categories = label_encoder.classes_
category_accuracy = {}
for i, cat in enumerate(categories):
    idxs = np.where(all_labels == i)[0]
    acc = accuracy_score(all_labels[idxs], all_preds[idxs]) if len(idxs) > 0 else 0
    category_accuracy[cat] = acc
pd.DataFrame([category_accuracy]).to_csv(os.path.join(output_path, "category_accuracy.csv"), index=False)
print("Category-wise Accuracy:", category_accuracy)

# ----------------------------
# Summary report
# ----------------------------
summary = metrics.copy()
summary.update(category_accuracy)
pd.DataFrame([summary]).to_csv(os.path.join(output_path, "summary_report.csv"), index=False)

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
plt.close()

# ----------------------------
# Confidence distribution per category (Violin + KDE)
# ----------------------------
# Violin plot
plt.figure(figsize=(12,6))
for i, cat in enumerate(categories):
    cat_probs = all_probs[all_labels == i, i]
    if len(cat_probs) > 0:
        sns.violinplot(
            x=[cat]*len(cat_probs),
            y=cat_probs,
            inner="quartile",
            hue=[cat]*len(cat_probs),
            palette="Set2",
            legend=False
        )
plt.ylabel("Confidence Score")
plt.title("Confidence Distribution per Category (Violin Plot)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "confidence_distribution_violin.png"))
plt.close()

# KDE plot
plt.figure(figsize=(12,6))
for i, cat in enumerate(categories):
    cat_probs = all_probs[all_labels == i, i]
    if len(cat_probs) > 0:
        sns.kdeplot(cat_probs, label=cat, fill=True)
plt.xlabel("Confidence Score")
plt.ylabel("Density")
plt.title("Confidence Distribution per Category (KDE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "confidence_distribution_kde.png"))
plt.close()

# ----------------------------
# Accuracy by category (Bar chart)
# ----------------------------
plt.figure(figsize=(10,6))
sns.barplot(
    x=list(category_accuracy.keys()),
    y=list(category_accuracy.values()),
    hue=list(category_accuracy.keys()),
    dodge=False,
    palette="viridis",
    legend=False
)
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Accuracy by Category")
for i, v in enumerate(category_accuracy.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "accuracy_by_category.png"))
plt.close()

# ----------------------------
# Overall metrics chart
# ----------------------------
plt.figure(figsize=(8,5))
overall_values = [overall_accuracy, f1, precision, recall, average_confidence]
overall_labels = ["Accuracy", "F1", "Precision", "Recall", "Avg Confidence"]
sns.barplot(
    x=overall_labels,
    y=overall_values,
    hue=overall_labels,
    dodge=False,
    palette="magma",
    legend=False
)
for i, v in enumerate(overall_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.ylim(0,1)
plt.title("Overall Metrics")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "overall_metrics_chart.png"))
plt.close()

# ----------------------------
# Save predictions
# ----------------------------
pred_labels = label_encoder.inverse_transform(all_preds)
true_labels = label_encoder.inverse_transform(all_labels)
pred_df = pd.DataFrame({
    "text": texts,
    "true_label": true_labels,
    "pred_label": pred_labels,
    "confidence": np.max(all_probs, axis=1)
})
pred_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

print(f"Evaluation complete. All outputs saved in {output_path}")
