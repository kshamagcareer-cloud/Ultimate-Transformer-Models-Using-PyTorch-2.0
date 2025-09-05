from transformers import BertTokenizer, BertForQuestionAnswering
from datasets import load_dataset

# Load pre-trained BERT tokenizer and model for extractive QA
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased")

# Load the SQuAD dataset
dataset = load_dataset("squad")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
