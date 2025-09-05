from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERTSUM model and tokenizer for extractive summarization
model = BertForSequenceClassification.from_pretrained("facebook/bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("facebook/bert-large-uncased-whole-word-masking-finetuned-squad")

# Tokenize the dataset for extractive summarization
def tokenize_extractive_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_extractive_datasets = dataset.map(tokenize_extractive_function, batched=True)

from transformers import Trainer, TrainingArguments

# Set up training arguments for extractive summarization
training_args = TrainingArguments(
    output_dir="./results_extractive",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_extractive_datasets["train"],
    eval_dataset=tokenized_extractive_datasets["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Generate an extractive summary
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
