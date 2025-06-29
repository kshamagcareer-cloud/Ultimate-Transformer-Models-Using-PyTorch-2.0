# Step 1: Install Necessary Libraries
# pip install torch torchaudio transformers

# Step 2: Load Pre-Trained Wav2Vec 2.0 Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()

# Step 3: Preprocess and Transcribe Audio
import torchaudio

# Load audio file
audio_path = "audio_file.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample audio to 16kHz if needed
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Process audio
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Generate transcription
with torch.no_grad():
    logits = model(input_values).logits

# Decode predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)
