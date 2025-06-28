# torchscript_export.py
# Demonstrates converting PyTorch models to TorchScript using both scripting and tracing,
# and enabling lazy tensor backend for optimization.

import torch
import torch._lazy  # For enabling lazy tensor backend

# === 1. Define a Simple Feedforward Model ===
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = torch.nn.Linear(10, 1)  # One linear layer: input 10 -> output 1

    def forward(self, x):
        return self.fc(x)

# === 2. Instantiate the model ===
model = SimpleNet()

# === 3. Convert the model to TorchScript using scripting ===
scripted_model = torch.jit.script(model)  # Best for models with control flow
scripted_model.save("simple_net.pt")  # Save the scripted model to a file

# === 4. Convert the model to TorchScript using tracing ===
# Tracing records the actual operations from an example input
example_inputs = torch.randn(1, 10)  # Dummy input to simulate one forward pass
traced_model = torch.jit.trace(model, example_inputs=example_inputs)

# Optional: Save traced model if needed
traced_model.save("simple_net_traced.pt")

# === 5. Enable Lazy Tensor Backend (experimental optimization) ===
torch._lazy.init()  # Initializes lazy tensor backend (may require LazyTensor-compatible hardware/runtime)
