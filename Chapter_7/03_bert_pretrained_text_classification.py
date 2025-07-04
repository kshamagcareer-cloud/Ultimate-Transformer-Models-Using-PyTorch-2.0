from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize input text
inputs = tokenizer("This is an example sentence.", return_tensors="pt", padding=True, truncation=True, max_length=512)

# Check tokenized input
print(inputs)
