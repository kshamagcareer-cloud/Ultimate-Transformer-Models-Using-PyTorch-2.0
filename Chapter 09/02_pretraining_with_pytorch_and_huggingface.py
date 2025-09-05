from datasets import load_dataset

# Load Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")
print(dataset)

from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import AutoModelForMaskedLM

# Load pre-trained BERT model for MLM training
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# If training a GPT-like model, use:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("gpt2")

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./bert_pretrained",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,  # Enable mixed precision training for speed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
