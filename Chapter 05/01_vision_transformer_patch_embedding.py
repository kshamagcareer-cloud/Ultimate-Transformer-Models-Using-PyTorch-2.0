import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # Input shape: [B, C, H, W] -> [B, embed_dim, n_patches_h, n_patches_w]
        x = self.proj(x)
        x = x.flatten(2)  # Flatten spatial dimensions
        x = x.transpose(1, 2)  # [B, N, embed_dim], N = number of patches
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # +1 for [CLS] token
        
    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        return x

from transformers import ViTForImageClassification, ViTConfig

# Configuration for ViT model
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_labels=10,  # Example for CIFAR-10
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

# Initialize Vision Transformer
model = ViTForImageClassification(config)

# Usage Example
img = torch.randn(1, 3, 224, 224)  # Simulate an image with batch size 1
patch_embed = PatchEmbedding()
patch_tokens = patch_embed(img)  # Output: [1, 196, 768] (196 patches, 768-dim embeddings)
