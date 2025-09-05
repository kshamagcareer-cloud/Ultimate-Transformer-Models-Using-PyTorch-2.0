# debugging_and_profiling.py
# Demonstrates various debugging, gradient inspection, profiling, and memory management tools in PyTorch

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity

# === 1. Define a simple model for testing ===
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model and data
model = SimpleNet()
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
loss_fn = nn.MSELoss()

# === 2. Check gradients after backpropagation ===
output = model(inputs)
loss = loss_fn(output, targets)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} - Gradients: {param.grad}")

# === 3. Register a forward hook to inspect layer output ===
def hook_fn(module, input, output):
    print(f"Layer: {module}")
    print(f"Output shape: {output.shape}")

model.fc1.register_forward_hook(hook_fn)  # Register hook on fc1
_ = model(inputs)  # Trigger the hook by running a forward pass

# === 4. Use anomaly detection to debug NaNs or Infs in backward pass ===
with torch.autograd.detect_anomaly():
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss.backward()

# === 5. Use torch.utils.benchmark to measure performance of forward pass ===
timer = benchmark.Timer(
    stmt="model(inputs)",
    globals={"model": model, "inputs": inputs}
)
print(timer.timeit(100))  # Run the statement 100 times and report avg time

# === 6. Profile detailed CPU and GPU operations using torch.profiler ===
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        _ = model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total"))

# === 7. Check GPU memory usage (if using CUDA) ===
if torch.cuda.is_available():
    print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
    print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")

    # Free up unused GPU memory
    torch.cuda.empty_cache()
