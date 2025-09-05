from transformers import AutoModelForSequenceClassification

# Load pre-trained text classification model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Tokenize and perform classification
inputs = tokenizer("I love using Hugging Face's Transformers!", return_tensors="pt")
outputs = model(**inputs)

# Extract predicted label
logits = outputs.logits
predicted_class = torch.argmax(logits).item()
print(f"Predicted class: {predicted_class}")
