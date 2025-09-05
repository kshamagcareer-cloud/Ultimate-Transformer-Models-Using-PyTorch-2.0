import requests

url = "http://localhost:8080/predictions/bert-text-classifier"
data = {"text": "This is an example sentence for classification."}

response = requests.post(url, json=data)

print(response.json())  # Output the model’s prediction
