from transformers import DetrForObjectDetection
from transformers import DetrImageProcessor
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Install required libraries
# pip install transformers torch torchvision

# Load pre-trained DETR model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()  # Set model to evaluation mode

# Load an image from the web
url = "https://t3.ftcdn.net/jpg/02/01/88/00/360_F_201880065_Ck9mn1J9NeIvbmwHBJCCG5vsuml6HYfK.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
inputs = processor(images=image, return_tensors="pt")

# Perform detection
outputs = model(**inputs)

# Extract bounding boxes and class predictions
logits = outputs.logits
bboxes = outputs.pred_boxes

# Map class labels to COCO dataset labels
COCO_CLASSES = processor.config.id2label

# Plot image and draw bounding boxes
def plot_results(image, outputs):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    logits = outputs.logits.softmax(-1)[0]
    bboxes = outputs.pred_boxes[0]
    
    # Draw bounding boxes
    for logit, bbox in zip(logits, bboxes):
        if logit.argmax() != 91:  # Skip "no object" class (index 91)
            score = logit.max().item()
            label = COCO_CLASSES[logit.argmax().item()]
            box = bbox.cpu().detach().numpy()
            
            x, y, w, h = box
            rect = patches.Rectangle(
                (x - w/2, y - h/2), w, h, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x, y, f"{label}: {score:.2f}", color="white", bbox=dict(facecolor="red", alpha=0.5))
    
    plt.show()

plot_results(image, outputs)
