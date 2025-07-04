from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fine-tuning BERT on a specific task, e.g., sentiment analysis
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", per_device_train_batch_size=8),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
