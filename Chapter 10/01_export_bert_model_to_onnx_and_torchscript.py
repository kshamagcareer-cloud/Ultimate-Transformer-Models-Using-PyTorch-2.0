import torch
from transformers import AutoModelForSequenceClassification

# Load a pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Set the model to evaluation mode
model.eval()

# Dummy input for export (batch_size, sequence_length)
dummy_input = torch.ones(1, 512, dtype=torch.long)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "bert_model.onnx", input_names=["input_ids"], output_names=["logits"])

# Example of Exporting a BERT Model to TorchScript Using Tracing
# Use tracing to convert the model to TorchScript
traced_model = torch.jit.trace(model, dummy_input)

# Save the TorchScript model
traced_model.save("bert_model_traced.pt")

# Example of Exporting a BERT Model to TorchScript Using Scripting
# Use scripting to convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the scripted model
scripted_model.save("bert_model_scripted.pt")
