# Using TorchScript to Optimize a Transformer Model
import torch

# Convert model to TorchScript format
traced_model = torch.jit.trace(model, torch.rand(1, 512))
torch.jit.save(traced_model, "optimized_model.pt")
print("TorchScript model saved for optimized execution.")

# Export model to ONNX format
import torch.onnx
torch.onnx.export(model, torch.rand(1, 512), "model.onnx")
print("Model exported to ONNX for optimized inference.")

# Running Models on GPUs with FP16 Mixed Precision
# Enable mixed precision
model.half()
model.to("cuda")

# Perform inference
with torch.no_grad():
    input_tensor = torch.rand(1, 512).half().to("cuda")
    output = model(input_tensor)

print("Inference completed with mixed precision.")

# Efficient Batch Processing for Large-Scale Inference
from transformers import AutoTokenizer

# Load tokenizer and process multiple texts
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["This is great!", "I'm not sure about this.", "Best movie ever!"]
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Move to GPU and perform inference
encoded_inputs = {k: v.to("cuda") for k, v in encoded_inputs.items()}
with torch.no_grad():
    outputs = model(**encoded_inputs)

print("Batch inference completed.")
