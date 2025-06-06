# custom_dataloader_example.py
# Demonstrates custom datasets, transformations, dataloaders, and built-in datasets in PyTorch

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import os

# === 1. Custom Dataset Definition ===
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = os.listdir(img_dir)  # List of image filenames

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)  # Open image file
        label = 1 if 'cat' in self.img_labels[idx] else 0  # Simple rule-based labeling

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label

# === 2. Define Transformations ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to 128x128
    transforms.RandomHorizontalFlip(),  # Apply horizontal flip randomly
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize pixel values
                         std=[0.229, 0.224, 0.225])
])

# === 3. Instantiate the Custom Dataset with Transforms ===
img_dir = "path/to/images"  # Replace with your actual image folder path
dataset = CustomImageDataset(img_dir, transform=transform)

# === 4. Use DataLoader to Load Data in Batches ===
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example loop through batches
for images, labels in dataloader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    # Here you would typically pass the batch to your model

# === 5. Using Built-in CIFAR-10 Dataset ===
# Define transformations specific to CIFAR-10
cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# Load CIFAR-10 training dataset
train_dataset = CIFAR10(root="data", train=True, transform=cifar_transform, download=True)

# Define DataLoader for CIFAR-10
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Example loop through CIFAR-10 batches
for images, labels in train_loader:
    print(f"CIFAR-10 batch shape: {images.shape}")
    # Feed into your model here
