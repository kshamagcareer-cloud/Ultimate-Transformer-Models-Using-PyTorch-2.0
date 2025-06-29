from transformers import ViTForImageClassification
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10  # For CIFAR-10, adjust for your dataset
)

# Preprocess the CIFAR-10 dataset by resizing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Download CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./vit_cifar10_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
)

# Use Hugging Face's Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Evaluate the model
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32)

# Evaluate model
accuracy = trainer.evaluate()
print(f"Test Accuracy: {accuracy['eval_accuracy']:.2f}")  # Example Output: Test Accuracy: 0.87

# Visualizing classification results
def show_predictions(model, dataloader):
    model.eval()
    images, labels = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs.logits, 1)

    fig, axs = plt.subplots(1, 5, figsize=(12, 4))
    for i in range(5):
        axs[i].imshow(np.transpose(images[i], (1, 2, 0)))
        axs[i].set_title(f"Pred: {preds[i].item()} | True: {labels[i]}")
        axs[i].axis('off')
    plt.show()

show_predictions(model, test_loader)
