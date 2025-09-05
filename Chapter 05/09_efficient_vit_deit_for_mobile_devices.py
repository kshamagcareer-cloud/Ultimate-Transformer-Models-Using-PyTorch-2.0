from transformers import DeiTForImageClassification
from transformers import DeiTFeatureExtractor
from PIL import Image
import requests
import torch

# Install necessary packages
# pip install transformers torch torchvision

# Load DeiT pre-trained on ImageNet
model = DeiTForImageClassification.from_pretrained("facebook/deit-small-distilled-patch16-224")
model.eval()

# Load an image
url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/COCO%20val2017%20000000036.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image
feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-small-distilled-patch16-224")
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform classification
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "deit_quantized.pth")
