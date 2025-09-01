Model Choices
The project uses BERT (bert-base-uncased) from HuggingFace’s Transformers library for sequence classification tasks.

The model has been adapted for multi-class classification to predict one of five predefined categories:

Billing Issue

Technical Problem

Compliment

Product Question

Complaint.

LabelEncoder from scikit-learn is used to transform text labels into integer ids for model training.

Fine-Tuning Method
The script loads and preprocesses the training data, encoding text labels and splitting data into train, validation, and test sets with stratification to maintain class balance.

Data is tokenized using the BertTokenizer with padding and truncation to a max length of 128 tokens.

BertForSequenceClassification is fine-tuned for 5 epochs, using:

Batch size: 8

Learning rate: 
2
×
10
−
5
2×10 
−5
 

Validation during training to select the best model based on weighted F1-score.

Trainer from HuggingFace is used for model training, validation, and metric calculation (accuracy, f1, precision, recall).

Evaluation Strategy
The final model is evaluated on a reserved test set, measuring:

Overall metrics: accuracy, weighted F1, precision, recall, average confidence, and total number of test cases.

Category-wise accuracy: accuracy per class.

Confusion Matrix: visualizing correct and incorrect predictions.

Confidence Distributions: per-category violin/kde plots of prediction confidence.

Bar charts compare accuracies and overall metrics visually.

All metrics and plots are saved to the Output directory for further analysis.

Deployment Instructions
Model Export:

Trained model, tokenizer, and label encoder are saved in classifier_model using HuggingFace and joblib.

API Deployment:

Flask serves the model as an HTTP API (see api.py).

The API loads model, tokenizer, and categories from the saved directory and exposes /classify for POST requests with input text.

The response includes predicted category and model confidence.

Run the server:

bash
python api.py
Access the web-based front-end at the default route (/). You can test classification via web UI or programmatically send requests to /classify.

Required Files and Environment:

Ensure trained model, tokenizer, and label encoder (classifier_model directory) exist.

Install dependencies:

text
pip install torch transformers scikit-learn flask pandas matplotlib seaborn joblib
This documentation ensures clarity about model selection, the fine-tuning pipeline, evaluation methodology, and the steps to deploy and run the classifier API, aligned with the project files and scripts.

