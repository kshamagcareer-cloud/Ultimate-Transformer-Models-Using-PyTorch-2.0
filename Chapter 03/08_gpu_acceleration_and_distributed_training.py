# gpu_acceleration_and_distributed_training.py
# Demonstrates GPU acceleration, multi-GPU data parallelism, and distributed training in PyTorch

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# === 1. Setup: Select device (GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Create tensors and move them to GPU ===
inputs = torch.randn(100, 10).to(device)
targets = torch.randn(100, 1).to(device)

# === 3. Define a simple model and move it to GPU ===
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet().to(device)

# === 4. Optional: Wrap model for data parallelism (multi-GPU on single machine) ===
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Automatically splits input across available GPUs

# Forward pass using DataParallel (if enabled above)
predictions = model(inputs)

# === 5. (Optional) Setup for Distributed Data Parallel (multi-machine or multi-process GPU) ===
# NOTE: The following code is for educational use.
# It requires launching the script with torchrun or torch.distributed.launch and passing rank/world_size/env variables.
# Uncomment and configure appropriately if running in a distributed environment.

"""
def setup_ddp(rank, world_size):
    # Initialize the distributed process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Move model to correct GPU and wrap it with DDP
    ddp_model = SimpleNet().to(rank)
    ddp_model = DDP(ddp_model, device_ids=[rank])
    return ddp_model
"""

# === 6. Check where tensors are located ===
print("Inputs located on:", inputs.device)
print("Model located on:", next(model.parameters()).device)
