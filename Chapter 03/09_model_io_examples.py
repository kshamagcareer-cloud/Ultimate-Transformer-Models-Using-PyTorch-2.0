# model_io_examples.py
# Demonstrates saving, loading, and checkpointing PyTorch models

import torch
import torch.nn as nn
import torch.optim as optim

# === 1. Define a simple model for demonstration ===
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Instantiate model and optimizer
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss = torch.tensor(0.0)  # Placeholder loss for demonstration
epoch = 5  # Example epoch

# === 2. Saving model parameters (state_dict) ===
torch.save(model.state_dict(), "model_weights.pth")

# === 3. Loading model parameters into a new instance ===
model_loaded = SimpleNet()
model_loaded.load_state_dict(torch.load("model_weights.pth"))

# === 4. Saving the entire model (architecture + parameters) ===
torch.save(model, "entire_model.pth")

# === 5. Loading the entire model (no need to redefine the class) ===
entire_model_loaded = torch.load("entire_model.pth")

# === 6. Saving a scripted TorchScript model for deployment ===
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

# === 7. Saving a training checkpoint (state_dict + optimizer state + metadata) ===
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "checkpoint.pth")

# === 8. Loading from a checkpoint to resume training ===
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Ensure model is in training mode after loading
model.train()
