# autograd_basics.py
# Demonstrates PyTorch's autograd functionality and gradient tracking

import torch

# === 1. Creating a tensor with gradient tracking enabled ===
x = torch.tensor([2.0, 3.0], requires_grad=True)
print("Input tensor x with requires_grad=True:\n", x)

# === 2. Performing operations on tensor ===
# y = x^2 + 3x + 1 (element-wise)
y = x**2 + 3 * x + 1
print("\nComputed tensor y from x:\n", y)

# === 3. Performing backpropagation ===
# Passing gradient of y w.r.t. output as [1.0, 1.0] to get gradients of x
y.backward(torch.tensor([1.0, 1.0]))

# === 4. Printing the gradient (dy/dx) ===
print("\nGradient of y w.r.t. x:\n", x.grad)  # Expected: [2*x + 3] => [7., 9.]

# === 5. Disabling gradient tracking for inference ===

# Method 1: Using torch.no_grad() context manager
with torch.no_grad():
    y_no_grad = x**2 + 3 * x + 1
    print("\nComputation with no_grad():\n", y_no_grad)

# Method 2: Detaching tensor from the computation graph
x_detached = x.detach()
print("\nDetached tensor (no gradient tracking):\n", x_detached)

# === 6. Gradient calculation in a simple linear model ===

# Simulated weights for a linear layer: shape (3 input features, 2 output features)
weights = torch.randn((3, 2), requires_grad=True)

# Input batch with shape (2 samples, 3 features)
inputs = torch.randn(2, 3)

# Forward pass: Matrix multiplication
output = inputs @ weights
print("\nOutput of simple linear model:\n", output)

# Loss function: just summing all outputs
loss = output.sum()

# Backward pass: compute gradients of weights w.r.t. loss
loss.backward()

# Print gradients of the weights
print("\nGradients of weights after backpropagation:\n", weights.grad)
