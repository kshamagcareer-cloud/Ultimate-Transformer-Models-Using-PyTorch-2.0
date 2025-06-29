from transformers import MaskFormerForInstanceSegmentation
from transformers import MaskFormerImageProcessor
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

# Install required packages
# pip install transformers torch torchvision

# Load pre-trained MaskFormer model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-coco")
model.eval()

# Load sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Black_Labrador_Retriever_-_Male_IMG_3323_%28cropped%29.jpg/1599px-Black_Labrador_Retriever_-_Male_IMG_3323_%28cropped%29.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-tiny-coco")
inputs = processor(images=image, return_tensors="pt")

# Perform segmentation
outputs = model(**inputs)

# Extract predicted masks and class labels
logits = outputs.class_queries_logits
masks = outputs.masks_queries_logits.sigmoid()

# Visualize segmentation results
def visualize_segmentation(image, outputs):
    masks = outputs.masks_queries_logits.sigmoid().cpu().detach()
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot masks
    for i in range(masks.shape[1]):
        plt.imshow(masks[0, i], cmap="jet", alpha=0.5)
    
    plt.title("Segmentation Overlay")
    plt.axis("off")
    plt.show()

visualize_segmentation(image, outputs)
