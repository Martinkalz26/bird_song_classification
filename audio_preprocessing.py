import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#waveform
aud="/content/drive/MyDrive/All1/major/CRYVAR08.mp3"


signal,sr =librosa.load(aud,sr=22050)
plt.figure(figsize=(10, 3))
librosa.display.waveshow(signal,sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveplot")
plt.show()

#Fourier Frequency Transform(FFT)

fft=np.fft.fft(signal)
magnitude=np.abs(fft)#for scaling
frequency=np.linspace(0, sr, len(magnitude))

lft_freq=frequency[:int(len(frequency)/2)]
lft_mag=magnitude[:int(len(frequency)/2)]

plt.figure(figsize=(10, 3))
plt.plot(lft_freq, lft_mag)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#stft
n_fft=2048
hop_length=512

stft=librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram=np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)#scaling

plt.figure(figsize=(10, 3))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length )
plt.xlabel("Time")
plt.xlabel("Frequency")
plt.colorbar()
plt.show()

# Load audio file and compute the MFCCs
audio_path = "/content/drive/MyDrive/All1/major/CRYVAR08.mp3"
y, sr = librosa.load(audio_path)
n_fft = 2048
hop_length = 512
MFCCs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# Create a new figure and set the size
plt.figure(figsize=(10, 3))  # Adjust the size as per your preference

# Display the MFCCs
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.title("MFCCs")
plt.show()