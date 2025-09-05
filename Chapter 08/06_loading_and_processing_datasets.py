from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Print dataset structure
print(dataset)

# Accessing and Manipulating Dataset Records
sample = dataset["train"][0]
print(sample)

batch = dataset["train"][:5]
print(batch)

texts = dataset["train"]["text"][:5]  # Extract first 5 text samples
labels = dataset["train"]["label"][:5]  # Extract first 5 labels
print(texts, labels)

# Transforming and Tokenizing Text Datasets
from transformers import AutoTokenizer

# Load tokenizer for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize dataset
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# View tokenized sample
print(tokenized_dataset["train"][0])

# Using Datasets with PyTorch DataLoaders
import torch
from torch.utils.data import DataLoader

# Set dataset format to PyTorch
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create PyTorch DataLoader
train_loader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=8)

# Streaming Large Datasets Without Full Loading
dataset = load_dataset("imdb", split="train", streaming=True)

# Iterate over the dataset in a memory-efficient way
for example in dataset:
    print(example)
    break  # Stop after one record to demonstrate streaming

# Saving and Sharing Processed Datasets
tokenized_dataset.save_to_disk("processed_dataset")

# To reload it later
from datasets import load_from_disk
dataset = load_from_disk("processed_dataset")

# To upload and share a dataset publicly
dataset.push_to_hub("my-dataset")
