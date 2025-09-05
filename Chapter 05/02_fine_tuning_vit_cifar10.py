from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import ViTForImageClassification, ViTConfig

# Data preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ViT input size
    transforms.ToTensor(),  # Convert images to tensors
])

# Download and load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Configure ViT for CIFAR-10
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_labels=10,  # CIFAR-10 has 10 classes
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

# Initialize ViT model
model = ViTForImageClassification(config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./vit_results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
