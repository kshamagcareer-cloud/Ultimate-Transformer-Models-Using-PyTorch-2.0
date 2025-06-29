from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")
model.eval()

# Step 3: Preprocess Audio Input
import torchaudio

# Load and preprocess an audio file
audio_path = "multi_lingual_audio.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample to 16kHz if necessary
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Prepare the input for the model
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Step 4: Generate Transcriptions
import torch

# Perform inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)
