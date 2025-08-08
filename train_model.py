import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# 1. Load dataset
df = pd.read_csv("data/malicious_phish.csv")
df = df.sample(n=500, random_state=42)  # Ubah n=... sesuai kebutuhan

# 2. Labeling: 1 = berbahaya, 0 = aman
df['label'] = df['type'].apply(lambda x: 1 if x != 'benign' else 0)

# 3. Preprocessing sederhana
df['url'] = df['url'].str.lower().str.replace(r'https?:\/\/', '', regex=True)
df['url'] = df['url'].str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)

# 4. Split train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['url'], df['label'], test_size=0.2, random_state=42)

# 5. Tokenisasi
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# 6. Buat dataset HuggingFace
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(train_labels)
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': list(test_labels)
})

# 7. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ✅ Tambahkan fungsi evaluasi metrik
def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    # predictions = torch.argmax(torch.tensor(logits), dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="backend/bert_model",          # ⬅ hasil model disimpan di sini
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",         # ⬅ harus cocok dengan compute_metrics
    greater_is_better=True
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics          # ✅ Wajib agar metrik evaluasi dihitung
)

# 10. Jalankan training
trainer.train()

# 11. Simpan model dan tokenizer
output_dir = "backend/bert_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)          
tokenizer.save_pretrained(output_dir)

# print("✅ Model berhasil dilatih dan disimpan di folder bert_model/")
print(f"✅ Model dan t.okenizer berhasil disimpan di: {output_dir}")