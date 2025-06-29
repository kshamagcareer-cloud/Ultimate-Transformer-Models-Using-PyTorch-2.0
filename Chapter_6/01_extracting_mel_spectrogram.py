import torchaudio
import torchaudio.transforms as T

# Load raw waveform
waveform, sample_rate = torchaudio.load("audio_file.wav")

# Resample if necessary
transform = T.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = transform(waveform)

# Compute mel-spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)
mel_spectrogram = mel_transform(waveform)

# Convert to log scale
log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
