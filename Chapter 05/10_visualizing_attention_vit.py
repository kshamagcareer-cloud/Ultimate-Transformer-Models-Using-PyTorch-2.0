from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load ViT model and feature extractor
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.eval()
processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load an image
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSh-as9CkzkyEKOJV8yi-dHdK0DeF1kH4hkwQ&s"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Perform forward pass to extract attention weights
outputs = model(**inputs, output_attentions=True)
attention = outputs.attentions[-1]  # Extract attention from the last layer

# Visualize Attention Maps
def plot_attention(image, attention):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    
    # Aggregate attention across all heads
    attention_map = attention[0].mean(dim=0).detach().numpy()
    
    # Resize attention to match the image size
    attention_map = attention_map.reshape(14, 14)  # ViT uses 14x14 patches
    attention_map = plt.imshow(attention_map, cmap='jet', alpha=0.5, extent=(0, image.size[0], image.size[1], 0))
    
    plt.colorbar(attention_map, ax=ax)
    plt.title("Attention Map Overlay")
    plt.axis("off")
    plt.show()

plot_attention(image, attention)
