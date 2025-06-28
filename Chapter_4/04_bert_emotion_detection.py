from datasets import load_dataset

# Load the GoEmotions dataset
dataset = load_dataset("go_emotions")

from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset for emotion detection
def tokenize_emotion_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization to the emotion dataset
tokenized_emotion_datasets = dataset.map(tokenize_emotion_function, batched=True)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model for multi-class emotion classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)  # Assume 28 emotions

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results_emotion",
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
    train_dataset=tokenized_emotion_datasets["train"],
    eval_dataset=tokenized_emotion_datasets["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the emotion detection task
trainer.evaluate()
