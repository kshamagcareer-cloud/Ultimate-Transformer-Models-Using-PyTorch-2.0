# nn_modules_examples.py
# Demonstrates custom, predefined, and nested module usage in PyTorch

import torch
import torch.nn as nn

# === 1. Creating a Custom Module ===

# Define a simple feedforward neural network with two fully connected layers
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # First fully connected layer: input_dim=10, hidden_dim=50
        self.fc2 = nn.Linear(50, 1)   # Output layer: hidden_dim=50, output_dim=1

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU after first layer
        x = self.fc2(x)              # Final linear transformation
        return x

# Instantiate and print the SimpleNet model
simple_model = SimpleNet()
print("SimpleNet architecture:\n", simple_model)


# === 2. Using Predefined Modules in PyTorch (CNN Example) ===

# Define a simple convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Conv layer: 3 input channels -> 16 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Max pooling: 2x2
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # Fully connected layer for classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 16 * 16 * 16)              # Flatten for FC layer
        x = self.fc1(x)                           # Output
        return x

# Instantiate and print the SimpleCNN model
cnn_model = SimpleCNN()
print("\nSimpleCNN architecture:\n", cnn_model)


# === 3. Building Complex Architectures with Nested Modules ===

# Define a reusable Block module: Linear -> BatchNorm -> ReLU
class Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.fc(x)))
        return x

# Define a complex model composed of multiple Blocks
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.block1 = Block(10, 50)
        self.block2 = Block(50, 30)
        self.fc = nn.Linear(30, 1)  # Final output layer

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x

# Instantiate and print the ComplexModel
complex_model = ComplexModel()
print("\nComplexModel architecture:\n", complex_model)
