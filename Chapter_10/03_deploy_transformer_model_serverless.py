import torch
from transformers import AutoModelForSequenceClassification

# Load a pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Convert model to TorchScript
model.eval()
dummy_input = torch.ones(1, 512, dtype=torch.long)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("bert_model.pt")
