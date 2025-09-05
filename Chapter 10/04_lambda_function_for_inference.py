import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def lambda_handler(event, context):
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('s3://my-bucket/bert_model.pt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Parse input text from the event
    input_text = event["text"]
    inputs = tokenizer(input_text, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Return the result
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': logits.argmax().item()})
    }
