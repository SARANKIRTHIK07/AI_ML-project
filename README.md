Model Choices

This project uses BERT (bert-base-uncased) from HuggingFace’s transformers library for sequence classification tasks.

Adapted for multi-class classification to predict one of five categories:

Billing Issue

Technical Problem

Compliment

Product Question

Complaint

LabelEncoder from scikit-learn is used to map text labels → integer IDs for training.

Fine-Tuning Method

Training data is preprocessed and split into train/validation/test sets with stratified sampling for class balance.

Tokenization is done with BertTokenizer, using padding and truncation (max_length=128).

BertForSequenceClassification is fine-tuned for 5 epochs with:

Batch size: 8

Learning rate: 2e-5

Training uses HuggingFace Trainer, with metrics: accuracy, F1, precision, recall.

The best model is selected based on weighted F1-score.

Evaluation Strategy

The final model is evaluated on a reserved test set, reporting:

Overall Metrics: accuracy, weighted F1, precision, recall, average confidence.

Category-wise Accuracy: per-class accuracy.

Confusion Matrix: for correct vs incorrect predictions.

Confidence Distributions: violin/KDE plots per category.

Bar Charts: comparing accuracies and metrics.

All metrics and plots are saved in the Output/ directory.

Deployment Instructions
1. Model Export

The trained model, tokenizer, and label encoder are saved into classifier_model/ using HuggingFace + Joblib.

⚠ Note on Large Files:
Due to GitHub’s file size limitations, the trained model folder (classifier_model/) is not uploaded.
Please run train.py to generate the model locally before using the API.
