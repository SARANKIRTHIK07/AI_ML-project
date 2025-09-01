import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\data\train.csv"
model_save_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\models\classifier_model"
os.makedirs(model_save_path, exist_ok=True)

df = pd.read_csv(data_path)

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
label2id = {label: idx for idx, label in id2label.items()}

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label_id"], random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["label_id"].tolist(),
    test_size=0.1,  # 10% of train as validation
    stratify=train_df["label_id"],
    random_state=42
)

train_df.to_csv(os.path.join(os.path.dirname(data_path), "train.csv"), index=False)
test_df.to_csv(os.path.join(os.path.dirname(data_path), "test.csv"), index=False)

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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./output/logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    logging_steps=10
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Test metrics:", test_metrics)

pd.DataFrame([test_metrics]).to_csv("./output/test_metrics.csv", index=False)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
joblib.dump(label_encoder, os.path.join(model_save_path, "label_encoder.pkl"))

torch.save(model.state_dict(), "model.pth")


print(f"Model and label encoder saved at {model_save_path}")

