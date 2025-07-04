from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing text
text = "Hugging Face makes NLP easy!"
tokens = tokenizer.tokenize(text)
print(tokens)

# Encoding Text into Token IDs
encoded_inputs = tokenizer(text, return_tensors="pt")
print(encoded_inputs)

# Decoding Tokens Back to Text
decoded_text = tokenizer.decode(encoded_inputs["input_ids"][0])
print(decoded_text)

# Optimizing Tokenization for Performance
# Padding and Truncation
batch = ["This is a long sentence.", "Short."]
encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
print(encoded_batch)

# Efficient Batching with Fast Tokenizers
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
