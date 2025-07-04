from transformers import AutoModel, AutoTokenizer

# Specify the model name
model_name = "bert-base-uncased"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print(f"Loaded model: {model_name}")  # Output: Loaded model: bert-base-uncased

# Example text input
text = "Hugging Face makes working with transformers simple and efficient!"

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")  # Convert to PyTorch tensors

print(inputs)

# Perform inference with pre-trained model
import torch

# Load pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Perform inference
with torch.no_grad():  # Disable gradient calculation for efficiency
    outputs = model(**inputs.to(device))

# Extract last hidden states (embeddings)
embeddings = outputs.last_hidden_state
print(embeddings.shape)
