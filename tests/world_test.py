from __future__ import division
import numpy as np
import scipy as sp
import scipy

import pyworld as pw
from pylab import *
import scipy.io.wavfile as wav


#%%
import librosa

ref_level_db=20
num_freq=1025
sample_rate=20000
frame_length_ms=50
frame_shift_ms=12.5
min_level_db=-100
num_mels=80
num_freq=1025
min_mel_freq=125
max_mel_freq=7600

power=1.5

def _stft_parameters():
  n_fft = (num_freq - 1) * 2
  hop_length = int(frame_shift_ms / 1000 * sample_rate)
  win_length = int(frame_length_ms / 1000 * sample_rate)
  return n_fft, hop_length, win_length

def spectrogram(y):
  D = _stft(y)
  S = _amp_to_db(np.abs(D)) - ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
  return _griffin_lim(S ** power)          # Reconstruct phase

def melspectrogram(y):
  D = _stft(y)
  S = _amp_to_db(_linear_to_mel(np.abs(D)))
  return _normalize(S)

def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)

_mel_basis = None


def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (num_freq - 1) * 2
  return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels,
    fmin=min_mel_freq, fmax=max_mel_freq)


def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db



#%%

path="/home/eugening/Neural/MachineLearning/Speech/TrainingData/LJSpeech-1.0/wavs/LJ001-0001.wav"
(fs,x) = wav.read(path)
x=x.astype(float)
f0, sp, ap = pw.wav2world(x, fs)

ap_coded=pw.code_aperiodicity(ap,fs)
sp_coded=pw.code_spectral_envelope(sp,fs,128)

#%%
y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
#%%

stft_x = _stft(x)
stft_y = _stft(y)

#%%

mel_x = melspectrogram(_normalize(x))
mel_y = melspectrogram(_normalize(y))