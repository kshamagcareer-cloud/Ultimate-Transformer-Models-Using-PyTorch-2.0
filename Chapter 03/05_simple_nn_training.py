# simple_nn_training.py
# Defines, trains, and evaluates a simple feedforward neural network using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

# === 1. Define a simple feedforward neural network ===
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Input layer -> 50 hidden units
        self.fc2 = nn.Linear(50, 20)  # Hidden layer -> 20 units
        self.fc3 = nn.Linear(20, 1)   # Output layer -> 1 unit (regression)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation on first layer
        x = torch.relu(self.fc2(x))  # ReLU activation on second layer
        x = self.fc3(x)              # Linear output
        return x

# Instantiate the model
model = SimpleNet()

# === 2. Define loss function and optimizer ===
loss_fn = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# === 3. Generate dummy training data ===
inputs = torch.randn(100, 10)   # 100 samples, 10 features
targets = torch.randn(100, 1)   # 100 corresponding target values

# === 4. Training loop ===
epochs = 100  # Number of full passes through the dataset

for epoch in range(epochs):
    # Forward pass: compute predictions and loss
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

    # Backward pass: compute gradients and update parameters
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update model parameters

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === 5. Evaluation without gradient tracking ===
with torch.no_grad():
    predictions = model(inputs)
    evaluation_loss = loss_fn(predictions, targets)
    print(f"\nEvaluation Loss: {evaluation_loss.item():.4f}")
