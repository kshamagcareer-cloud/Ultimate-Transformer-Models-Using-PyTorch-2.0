from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fine-tuning the model
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs', 
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset
)

trainer.train()
