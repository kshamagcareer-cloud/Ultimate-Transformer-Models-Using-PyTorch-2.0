import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_scheduler
from sklearn.metrics import accuracy_score

# Define TextDataset for tokenization and loading data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Set up tokenizer, model, and dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_texts = ["Example text 1", "Example text 2"]  # Replace with actual text
train_labels = [0, 1]  # Replace with actual labels
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load pre-trained model (example)
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Get input data
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Learning rate scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * epochs,
)

# Evaluate model
test_preds = []
test_labels = []
model.eval()
for batch in test_dataloader:
    with torch.no_grad():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
