from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments

# Load pre-trained GPT-2 model with classification head
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# Tokenize and prepare dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Define training arguments
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)

# Initialize Trainer and train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
