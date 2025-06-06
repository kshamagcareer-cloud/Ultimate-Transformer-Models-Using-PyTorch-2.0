# tensor_basics.py
# Demonstrates basic PyTorch tensor creation and operations

import torch

# 1. Creating a tensor with specific values
tensor = torch.tensor([[1, 2], [3, 4]])  # 2x2 tensor with specific integers
print("Tensor with specific values:\n", tensor)

# 2. Creating a tensor with random values between 0 and 1
random_tensor = torch.rand(3, 3)  # 3x3 tensor with random float values
print("\nRandom tensor:\n", random_tensor)

# 3. Creating tensors filled with zeros and ones
zeros_tensor = torch.zeros(2, 2)  # 2x2 tensor of all zeros
ones_tensor = torch.ones(4, 4)    # 4x4 tensor of all ones
print("\nZeros tensor:\n", zeros_tensor)
print("Ones tensor:\n", ones_tensor)

# 4. Creating an empty (uninitialized) tensor with a given shape
shape = (2, 3)
empty_tensor = torch.empty(shape)
print("\nEmpty tensor with shape (2, 3):\n", empty_tensor)

# === Basic Tensor Operations ===

# Element-wise arithmetic operations
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
print("\nElement-wise addition:", x + y)
print("Element-wise subtraction:", x - y)
print("Element-wise multiplication:", x * y)
print("Element-wise division:", x / y)

# Matrix multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
matmul_result = torch.matmul(a, b)  # or a @ b
print("\nMatrix multiplication result:\n", matmul_result)

# Reshaping a tensor
tensor = torch.rand(2, 4)  # 2x4 tensor
reshaped_tensor = tensor.view(4, 2)  # Reshaped to 4x2
print("\nOriginal tensor shape (2,4):\n", tensor)
print("Reshaped tensor shape (4,2):\n", reshaped_tensor)

# Concatenating and stacking tensors
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
concatenated = torch.cat((x, y), dim=0)  # Concatenate along rows (dim=0)
stacked = torch.stack((x, y), dim=0)     # Stack along a new dim (adds extra dim)
print("\nConcatenated tensor:\n", concatenated)
print("Stacked tensor:\n", stacked)

# === Moving Tensors to GPU if available ===
tensor = torch.rand(3, 3)
if torch.cuda.is_available():
    tensor = tensor.to('cuda')  # Move tensor to GPU
    print("\nTensor moved to GPU.")
else:
    print("\nCUDA not available. Tensor remains on CPU.")
