# Step 0: Install transformers if not installed
# pip install transformers torch pandas

#For data maniplation and analysis, allows for us to read the CSV file and handle dataframes.
import pandas as pd
#Has Tensors (multi-dimensional arrays for gpus), has autograd for automatic differentiation for training 
import torch
#Dataset for making our custom dataset class and DataLoader for batching and shuffling data.
from torch.utils.data import Dataset, DataLoader
#Hugging Face library for pre-trained models and tokenizers.
from transformers import BertTokenizer, BertForSequenceClassification
#AdamW is an optimzer for weight decay
from torch.optim import AdamW
# ------------------------------
# Step 1: Load CSV and combine subject + text
# ------------------------------
df = pd.read_csv("dataset.csv")
# Combine subject and text for classification
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
#TRaining in batches of 1, shuffling the data for randomness.
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------------
# Step 4: Load BERT model for classification
# ------------------------------
#It’s a pre-trained language model from Google (2018) that understands text contextually. It reads text left -> right and right -> left 
#Num of labels is 2 for binary classification (pass/rejection).
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
#Selects the device to run the model on, either GPU (cuda) if available or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Step 5: Training setup
# ------------------------------
#Learning rate = 0.00002
#Determines how big each weight update is. Small learning rates are common for fine-tuning BERT, because pre-trained weights are already good — large updates could “break” them.
optimizer = AdamW(model.parameters(), lr=2e-5)

# Number of training "sections / passes" through the entire dataset.
epochs = 8

# ------------------------------
# Step 6: Training loop
# ------------------------------
#Puts model in training mode
model.train()
#For each epoch, iterate through the dataloader
for epoch in range(epochs):
    #For monitering purposes
    total_loss = 0
    #For every data in dataset
    for batch in dataloader:
        #Move tensor to the selected device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        
        #Make a forward pass, calcs loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        total_loss += loss.item()
        
        #Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# ------------------------------
# Step 7: Make prediction on new email
# ------------------------------
def classify_email(subject, text):
    # Set the model to evaluation mode (no dropout, no gradient updates)
    model.eval()
    # Combine the email subject and text into one string
    combined = subject + " " + text
    
    # Tokenize the text into numerical input for the model
    # - truncation=True cuts off extra-long text
    # - padding='max_length' ensures all inputs have the same length
    # - max_length=128 limits sequence length
    # - return_tensors='pt' returns PyTorch tensors instead of lists
    encoding = tokenizer(combined, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    
    # Move tokenized inputs to the GPU (or CPU) device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Turn off gradient calculation (for faster inference)
    with torch.no_grad():
        # Run the model to get predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply softmax to convert raw logits into probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Get the index (0 or 1) of the highest probability class
        pred = torch.argmax(probs, dim=-1).item()
        # Get the confidence score (probability of that predicted class)
        confidence = probs[0][pred].item()
    
    return pred, confidence

# Example usage
subject = "Interview Invitation - Software Engineer"
text = "Dear Ishan, we are excited to invite you for an interview next week."
pred, conf = classify_email(subject, text)
print(f"Prediction: {'Pass' if pred==1 else 'Rejection'} ({conf*100:.1f}% confidence)")
