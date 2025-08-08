import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# 🔄 Muat model & tokenizer
MODEL_DIR = "backend/bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 📄 Load data uji
df = pd.read_csv("data/data_uji.csv")  # Ganti sesuai nama file kamu
if 'url' not in df.columns:
    raise ValueError("Kolom 'url' tidak ditemukan dalam file.")

# 🧹 Preprocessing
def clean_url(url):
    url = str(url).lower()
    url = re.sub(r'https?:\/\/', '', url)
    url = re.sub(r'[^a-zA-Z0-9]', ' ', url)
    return url

df['clean_url'] = df['url'].apply(clean_url)

# 🔢 Tokenisasi
encodings = tokenizer(list(df['clean_url']), truncation=True, padding=True, max_length=128, return_tensors="pt")

# 🔍 Prediksi
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# 🧮 Statistik
df['prediction'] = predictions.numpy()
total = len(df)
benign = (df['prediction'] == 0).sum()
malicious = (df['prediction'] == 1).sum()

# 📊 Tampilkan hasil
print("\n📊 HASIL PREDIKSI:")
print("────────────────────────────")
print(f"🔐 Total URL: {total}")
print(f"✅ Aman (benign): {benign} ({(benign/total)*100:.2f}%)")
print(f"⚠️ Berbahaya (phishing/malware/defacement): {malicious} ({(malicious/total)*100:.2f}%)")
print("────────────────────────────")

# # 💾 (Opsional) Simpan hasil ke file
# df.to_csv("data/hasil_prediksi.csv", index=False)
