import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

MODEL_DIR = "backend/bert_model"

print("🔄 Memuat model dan tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

print("📥 Memuat dataset latih...")
df = pd.read_csv("data/malicious_phish.csv")
df = df.sample(n=5000, random_state=42)  # Atur jumlah sesuai kebutuhan

df['label'] = df['type'].apply(lambda x: 1 if x != 'benign' else 0)
df['url'] = df['url'].str.lower().str.replace(r'https?:\/\/', '', regex=True)
df['url'] = df['url'].str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['url'], df['label'], test_size=0.2, random_state=42
)

test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': list(test_labels)
})

def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

training_args = TrainingArguments(
    output_dir="temp_eval",  # folder sementara
    per_device_eval_batch_size=64,
    do_eval=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🔍 Melakukan evaluasi pada data latih/test split...")
eval_results = trainer.evaluate()

print("\n📊 Hasil Evaluasi:")
for key in ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']:
    if key in eval_results:
        print(f"{key}: {eval_results[key] * 100:.2f}%")