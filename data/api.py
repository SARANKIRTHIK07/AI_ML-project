from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

model_save_path = r"C:\Users\user\Desktop\AI project\data\pythonProject\models\classifier_model"

tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path, num_labels=5)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

categories = ["Billing Issue", "Technical Problem", "Compliment", "Product Question", "Complaint"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data["text"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return jsonify({
        "category": categories[predicted.item()],
        "confidence": round(confidence.item(), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
