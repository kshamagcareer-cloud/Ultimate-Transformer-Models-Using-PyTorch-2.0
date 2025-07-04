from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

import torch
from torch.utils.data import DataLoader

# Set dataset format for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define DataLoaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
test_loader = DataLoader(tokenized_datasets["test"], batch_size=8)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start fine-tuning
trainer.train()

from transformers import get_scheduler

# Use Learning Rate Scheduling
lr_scheduler = get_scheduler(
    name="linear", optimizer=trainer.optimizer, num_warmup_steps=500, num_training_steps=10000
)

from transformers import EarlyStoppingCallback

# Implement Early Stopping
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# Regularization Techniques
training_args.weight_decay = 0.01  # Applies L2 regularization
