import torch
import torch.nn as nn

# PatchEmbedding class for Vision Transformer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # Flatten spatial dimensions
        x = x.transpose(1, 2)  # [B, N, embed_dim], N = number of patches
        return x

# PositionalEncoding class for Vision Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # +1 for CLS token

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        return x

# Initialize CLS token (classification token)
cls_token = nn.Parameter(torch.randn(1, 1, 768))  # Learnable CLS token

# Transformer Encoder layer using ViTModel from HuggingFace
from transformers import ViTModel
transformer_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

# End-to-End ViT Model Configuration for Image Classification
from transformers import ViTForImageClassification, ViTConfig

# Configure ViT model
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_labels=10,  # Example for CIFAR-10
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

# Initialize ViT model for classification
model = ViTForImageClassification(config)
