from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()

import torchaudio

# Load an audio file
waveform, sample_rate = torchaudio.load("audio_file.wav")

# Resample to 16kHz if necessary
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Prepare input for the model
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Generate predictions
with torch.no_grad():
    logits = model(input_values).logits

# Decode the predicted IDs to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)
