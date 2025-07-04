from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize an input sentence
inputs = tokenizer("Hugging Face makes AI development easy!", return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
print(logits)
