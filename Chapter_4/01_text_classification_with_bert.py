from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")

from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import BertForSequenceClassification

# Load pre-trained BERT model for sequence classification (binary classification)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import CrossEntropyLoss

# Prepare the DataLoader
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the loss function
loss_fn = CrossEntropyLoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Training for 3 epochs
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        # Move batch to the same device as model
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

from sklearn.metrics import accuracy_score

# Prepare validation DataLoader
val_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
