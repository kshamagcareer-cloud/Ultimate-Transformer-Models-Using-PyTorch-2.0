import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset

# Step 1: Load dataset
dataset = load_dataset("imdb")  # Example dataset for sentiment classification

# Step 2: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Step 3: Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = dataset["train"].map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Step 4: DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Step 5: Model (DistilBERT for sequence classification)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 6: Loss Function
loss_fn = CrossEntropyLoss()

# Step 7: Optimizer (AdamW)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 8: Gradient Accumulation and Mixed Precision Training
accumulation_steps = 4  # Accumulate gradients over 4 steps
scaler = GradScaler()  # For mixed precision

# Training loop
optimizer.zero_grad()  # Reset gradients before starting training
for step, batch in enumerate(train_loader):
    inputs, labels = batch["input_ids"], batch["label"]
    outputs = model(inputs)
    loss = loss_fn(outputs.logits, labels)

    # Normalize loss to account for accumulation
    loss = loss / accumulation_steps
    loss.backward()  # Compute gradients

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients for next accumulation

# Mixed Precision Training
for batch in train_loader:
    inputs, labels = batch["input_ids"], batch["label"]
    optimizer.zero_grad()
    with autocast():  # Enables FP16 computations where possible
        outputs = model(inputs)
        loss = loss_fn(outputs.logits, labels)

    scaler.scale(loss).backward()  # Scale gradients to prevent underflow
    scaler.step(optimizer)  # Update weights
    scaler.update()  # Adjust scaling for next step
