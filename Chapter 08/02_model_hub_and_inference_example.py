from transformers import pipeline

# List all available models for a specific task
task = "text-classification"
models = pipeline(task).model.config._name_or_path
print(f"Pre-trained model for {task}: {models}")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained BERT model and tokenizer from the Hugging Face Model Hub
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input text
text = "Hugging Face makes working with AI seamless!"
inputs = tokenizer(text, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
print(outputs.logits)
