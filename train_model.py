import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os

import pandas as pd

# === Load CSV from Google Sheet ===
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf2Tm2H0Yvqs7-g5n_ysK0QYd0mVhhPKArdd7s-Z06mKd7UV4fjOJjbgUVODhqmXpk4_-OQHdyEnjn/pub?output=csv"
df = pd.read_csv(csv_url).dropna()

texts = df['text'].tolist()
labels = df['intent'].tolist()

prompt_examples = [{"text": t, "intent": i} for t, i in zip(texts, labels)]

# ========== STEP 3: Save prompt examples for LLaMA few-shot classification ==========
with open("prompt_training_examples.json", "w") as f:
    json.dump(prompt_examples, f, indent=2)
print("✅ Saved prompt examples for LLaMA to 'prompt_training_examples.json'")

# ========== STEP 4: Encode labels ==========
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)
np.save("label_classes.npy", label_encoder.classes_)

# ========== STEP 5: Tokenize ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

dataset = IntentDataset(texts, label_ids)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ========== STEP 6: Train BERT ==========
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(4):
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ========== STEP 7: Save BERT model ==========
model.save_pretrained("intent_model")
tokenizer.save_pretrained("intent_model")
print("✅ BERT model saved to 'intent_model/'")
