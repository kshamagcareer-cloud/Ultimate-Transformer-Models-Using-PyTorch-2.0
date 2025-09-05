import torch
from transformers import AutoModelForSequenceClassification

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_model.pth")
print("Model quantized and ready for fast inference.")

from transformers import AutoModelForSequenceClassification

# Load DistilBERT instead of full BERT
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
print("Loaded lightweight DistilBERT model for efficient inference.")

