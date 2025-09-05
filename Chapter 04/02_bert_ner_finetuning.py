from datasets import load_dataset

# Load the CoNLL-03 dataset
dataset = load_dataset("conll2003")

from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)
    labels = examples["ner_tags"]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

from transformers import BertForTokenClassification, Trainer, TrainingArguments

# Load pre-trained BERT for token classification
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)

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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

from sklearn.metrics import classification_report

# Evaluate the model
predictions, true_labels, _ = trainer.predict(tokenized_datasets["validation"])

# Convert predictions and true labels
pred_labels = predictions.argmax(axis=-1)

# Flatten arrays for classification report
pred_labels_flat = pred_labels.flatten()
true_labels_flat = true_labels.flatten()

# Generate classification report
print(classification_report(true_labels_flat, pred_labels_flat))
