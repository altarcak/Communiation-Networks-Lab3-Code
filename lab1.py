import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
from scipy import io
from scipy import signal
import random
import makelab
from makelab import signal
from makelab import audio



sampling_rate, audio_data = sp.io.wavfile.read('data/audio/HumanVoice-Test_16bit_44.1kHz_mono.wav')
quantization_bits = 16


if len(audio_data.shape) == 2:
    audio_data = audio_data.mean(axis=1) 


factor = 10
audio_down = audio_data[::factor]
sr_down = sampling_rate // factor

xlim_zoom = (11500, 12500)
resample_xlim_zoom = (xlim_zoom[0] / factor, xlim_zoom[1] / factor)

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4)

axs[0, 0].plot(audio_data)
axs[0, 0].set_title("Original Signal")
axs[0, 0].axvspan(xlim_zoom[0], xlim_zoom[1], color='red', alpha=0.3)


axs[0, 1].plot(np.arange(xlim_zoom[0], xlim_zoom[1]), audio_data[xlim_zoom[0]:xlim_zoom[1]])
axs[0, 1].set_title(f"Original Signal (Zoom: {xlim_zoom})")

x = np.arange(int(resample_xlim_zoom[0]), int(resample_xlim_zoom[1]))
y = audio_down[int(resample_xlim_zoom[0]):int(resample_xlim_zoom[1])]

axs[1, 0].stem(x, y, basefmt=" ")
axs[1, 0].set_title("Downsampled Signal")

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_down.astype(float))), ref=np.max)
img = librosa.display.specshow(D, sr=sr_down, x_axis='time', y_axis='hz', ax=axs[1, 1])
axs[1, 1].set_title("Downsampled Spectrogram ")
fig.colorbar(img, ax=axs[1, 1], format="%+2.f dB")

nyquist_or = sampling_rate / 2
nyquist_down = sr_down / 2
print(f"Original sr: {sampling_rate} Hz, Nyquist: {nyquist_or} Hz")
print(f"New sr: {sr_down} Hz, New Nyquist: {nyquist_down} Hz")


print("Original Sound:")
ipd.display(ipd.Audio(audio_data, rate=sampling_rate))



print("Aliased Sound:")
ipd.display(ipd.Audio(audio_down, rate=sr_down))

from scipy.signal import chirp, butter, lfilter


sr = 44100  
duration = 5       
t = np.linspace(0, duration, duration * sr)


sweep_signal = chirp(t, f0=0, f1=22000, t1=duration, method='linear')

factor = 10
sr_down = sr // factor
sweep_down = sweep_signal[::factor]


nyquist_down = sr_down / 2

b, a = butter(N=8, Wn=nyquist_down, fs=sr, btype='low')
filtered_signal = lfilter(b, a, sweep_signal)
sweep_down_filtered = filtered_signal[::factor]


fig, axs = plt.subplots(1, 3, figsize=(18, 5))


D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(sweep_signal.astype(float))), ref=np.max)
librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=axs[0])
axs[0].set_title("Original Sweep")


D_alias = librosa.amplitude_to_db(np.abs(librosa.stft(sweep_down.astype(float))), ref=np.max)
librosa.display.specshow(D_alias, sr=sr_down, x_axis='time', y_axis='hz', ax=axs[1])
axs[1].set_title("Aliased Sweep")


D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(sweep_down_filtered.astype(float))), ref=np.max)
librosa.display.specshow(D_filtered, sr=sr_down, x_axis='time', y_axis='hz', ax=axs[2])
axs[2].set_title("Filtred Downsample Sweep")

print("Original Sweep Signal:")
ipd.display(ipd.Audio(sweep_signal, rate=sr))


print("Aliased Sweep Signal:")
ipd.display(ipd.Audio(sweep_down, rate=sr_down))


print("Filtred Downsample Sound:")
ipd.display(ipd.Audio(sweep_down_filtered, rate=sr_down))



plt.tight_layout()
plt.show()
