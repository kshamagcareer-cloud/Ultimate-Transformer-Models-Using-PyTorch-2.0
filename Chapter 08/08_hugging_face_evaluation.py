import evaluate

# Load the accuracy metric
accuracy = evaluate.load("accuracy")

# Sample predictions and labels
predictions = [0, 1, 1, 0]
references = [0, 1, 0, 0]

# Compute accuracy
results = accuracy.compute(predictions=predictions, references=references)
print(results)

from transformers import pipeline

# Load a fine-tuned sentiment classification model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Test model performance
results = classifier(["I love this product!", "This was a terrible experience."])
print(results)

# Load models
bert_classifier = pipeline("text-classification", model="bert-base-uncased")
roberta_classifier = pipeline("text-classification", model="roberta-base")

# Define test sentence
test_text = "Hugging Face makes AI easy to use!"

# Evaluate both models
bert_result = bert_classifier(test_text)
roberta_result = roberta_classifier(test_text)

print(f"BERT Prediction: {bert_result}")
print(f"RoBERTa Prediction: {roberta_result}")

from datasets import load_dataset
from transformers import pipeline

# Load a dataset
dataset = load_dataset("imdb", split="test[:100]")  # Load a sample of 100 reviews

# Load fine-tuned classifier
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Get model predictions
predictions = [classifier(review["text"])[0]["label"] for review in dataset]

# Convert labels to binary (POSITIVE → 1, NEGATIVE → 0)
binary_predictions = [1 if pred == "POSITIVE" else 0 for pred in predictions]
binary_labels = [label for label in dataset["label"]]

# Compute accuracy
accuracy = evaluate.load("accuracy").compute(predictions=binary_predictions, references=binary_labels)
print(f"Test Accuracy: {accuracy['accuracy']:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(binary_labels, binary_predictions)

# Plot heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Sentiment Analysis")
plt.show()
