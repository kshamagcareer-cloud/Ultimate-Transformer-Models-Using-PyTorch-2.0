from datasets import load_dataset

# Load a popular dataset (IMDB sentiment analysis dataset)
dataset = load_dataset("imdb")

# Print dataset structure
print(dataset)

import re

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sample text
text = "Transformers are revolutionizing natural language processing!"
tokens = tokenizer(text)
print(tokens)

# Apply tokenization with padding and truncation
tokenized_data = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

import torch
from torch.utils.data import DataLoader

# Convert dataset to PyTorch format
tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define DataLoader
train_loader = DataLoader(tokenized_data["train"], batch_size=8, shuffle=True)
test_loader = DataLoader(tokenized_data["test"], batch_size=8)

# Iterate over the DataLoader
for batch in train_loader:
    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["label"]
    print(input_ids.shape, attention_mask.shape, labels.shape)
    break

# Using Streaming for Large Datasets
dataset = load_dataset("c4", split="train", streaming=True)

# Example of streaming and breaking after one record
for example in dataset:
    print(example)
    break  # Stop after first record to demonstrate streaming

# Sharding Data for Distributed Training
train_data = tokenized_data["train"].shard(num_shards=10, index=0)  # Use only 1/10th of dataset
