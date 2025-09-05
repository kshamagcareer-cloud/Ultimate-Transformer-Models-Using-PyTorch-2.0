import torchaudio
from transformers import Wav2Vec2Processor

# Load audio file
audio_path = "path_to_audio_file.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample audio to 16 kHz (required for Wav2Vec 2.0)
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Normalize the waveform
waveform = waveform / waveform.abs().max()

# Initialize Wav2Vec 2.0 processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Load the pre-trained Wav2Vec 2.0 model
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()

# Perform inference
import torch
with torch.no_grad():
    logits = model(input_values).logits

# Decode the predicted IDs into text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)

# Clean up the transcription (e.g., punctuation, case normalization)
def clean_transcription(text):
    return text.strip().capitalize()

cleaned_transcription = clean_transcription(transcription)
print("Cleaned Transcription:", cleaned_transcription)
