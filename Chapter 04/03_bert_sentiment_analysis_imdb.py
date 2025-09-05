from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model for binary sentiment classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
