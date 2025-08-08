from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==== MODEL SETUP ====
MODEL_PATH = "backend/bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# ==== PREPROCESSING ====
def preprocess_url(url):
    url = url.lower()
    url = re.sub(r"https?:\/\/", "", url)
    url = re.sub(r"[^a-zA-Z0-9]", " ", url)
    return url.strip()

# ==== PREDICTION FUNCTION ====
def predict_url(url):
    clean_url = preprocess_url(url)
    inputs = tokenizer(clean_url, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()
    return {
        "label": "malicious" if prediction == 1 else "benign",
        "confidence": float(confidence)
    }

# ==== API ROUTE ====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = predict_url(url)
    return jsonify(result)

# ==== MAIN ====
if __name__ == '__main__':
    app.run(debug=True)
