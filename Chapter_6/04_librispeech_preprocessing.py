from datasets import load_dataset

# Load a labeled dataset
dataset = load_dataset("librispeech_asr", split="train.100")

# Tokenize the dataset
def preprocess(batch):
    audio = batch["audio"]
    input_values = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values[0]
    with processor.as_target_processor():
        labels = processor(batch["text"], return_tensors="pt").input_ids[0]
    return {"input_values": input_values, "labels": labels}

dataset = dataset.map(preprocess)
