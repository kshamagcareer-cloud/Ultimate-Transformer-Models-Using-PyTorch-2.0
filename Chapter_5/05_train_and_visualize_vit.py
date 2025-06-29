from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt

# Data preprocessing (CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Download and load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./vit_results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Function to visualize attention maps
def visualize_attention(model, img):
    model.eval()
    with torch.no_grad():
        outputs = model(img.unsqueeze(0))  # Unsqueeze for batch dimension
        attentions = model.vit.encoder.layer[-1].attention.attention_probs  # Last layer attention
    attn = attentions.mean(1).squeeze(0).cpu().detach().numpy()
    plt.imshow(attn[0], cmap='hot')
    plt.title("Attention Map")
    plt.show()
