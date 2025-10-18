# Step 0: Install transformers if not installed
# pip install transformers torch pandas

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
# ------------------------------
# Step 1: Load CSV and combine subject + text
# ------------------------------
df = pd.read_csv("dataset.csv")
texts = (df['subject'] + " " + df['text']).tolist()
labels = df['label'].tolist()

# ------------------------------
# Step 2: Create PyTorch Dataset
# ------------------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # shape [max_len]
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ------------------------------
# Step 3: Initialize tokenizer and dataset
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = EmailDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------------
# Step 4: Load BERT model for classification
# ------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Step 5: Training setup
# ------------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

# ------------------------------
# Step 6: Training loop
# ------------------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# ------------------------------
# Step 7: Make prediction on new email
# ------------------------------
def classify_email(subject, text):
    model.eval()
    combined = subject + " " + text
    encoding = tokenizer(combined, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence

# Example usage
subject = "Interview Invitation - Software Engineer"
text = "Dear Ishan, we are excited to invite you for an interview next week."
pred, conf = classify_email(subject, text)
print(f"Prediction: {'Pass' if pred==1 else 'Rejection'} ({conf*100:.1f}% confidence)")
