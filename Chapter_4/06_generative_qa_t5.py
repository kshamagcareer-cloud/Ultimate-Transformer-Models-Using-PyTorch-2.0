from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Example input
context = "The Eiffel Tower is located in Paris, France."
question = "Where is the Eiffel Tower located?"

# Preprocess the input
input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

# Generate answer
outputs = model.generate(inputs["input_ids"], max_length=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)  # Output: Paris, France
