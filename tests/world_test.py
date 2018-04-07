from __future__ import division
import numpy as np
import scipy as sp
import scipy

import pyworld as pw
from pylab import *
import scipy.io.wavfile as wav

import simpleaudio as sa
import librosa


#%%

ref_level_db=20
num_freq=1025

frame_length_ms=5
frame_shift_ms=1.25
min_level_db=-100
num_mels=80
num_freq=1025
min_mel_freq=125
max_mel_freq=7600

power=1.5

def _stft_parameters(sample_rate):
  n_fft = (num_freq - 1) * 2
  hop_length = int(frame_shift_ms / 1000 * sample_rate)
  win_length = int(frame_length_ms / 1000 * sample_rate)
  return n_fft, hop_length, win_length

def spectrogram(y, fs):
  D = _stft(y, fs)
  S = _amp_to_db(np.abs(D)) - ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
  return _griffin_lim(S ** power)          # Reconstruct phase

def melspectrogram(y, fs):
  D = _stft(y, fs)
  S = _amp_to_db(_linear_to_mel(np.abs(D)))
  return _normalize(S)

def _stft(y, fs):
  n_fft, hop_length, win_length = _stft_parameters(fs)
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, fs):
  _, hop_length, win_length = _stft_parameters(fs)
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

stft_x = _stft(x, fs)
stft_y = _stft(y, fs)

#%%

mel_x = melspectrogram(_normalize(x))
mel_y = melspectrogram(_normalize(y))

#%%

def encode(x, fs, frame_period):
    x=x.astype(float)
    f0, sp, ap = pw.wav2world(x, fs, frame_period)
    
    nframes = int(fs * f0.size * frame_period/1000)
    x = np.pad(x,(0,nframes-x.size), 'constant')

    f01, sp1, ap1 = pw.wav2world(x, fs, frame_period)
    w_x=np.hstack([f01[:,np.newaxis],sp1,ap1])

    y = pw.synthesize(f0*(1+0*np.random.normal(0,.05,size=f0.shape)), 
                      sp*(1+1*np.random.normal(0,.2,size=sp.shape)), 
                      ap*(1+1*np.random.normal(0,.2,size=ap.shape)), fs, frame_period)

    f02, sp2, ap2 = pw.wav2world(y, fs, frame_period)
    w_y = np.hstack([f02[:,np.newaxis],sp2,ap2])
    
    stft_x = np.abs(_stft(x, fs))
    stft_y = np.abs(_stft(y, fs))
    
    return (x,y,f0,sp,ap,stft_x,stft_y,w_x,w_y)
    

path="/home/eugening/Neural/MachineLearning/Speech/TrainingData/LJSpeech-1.0/wavs/LJ001-0001.wav"
(fs,x) = wav.read(path)

(x,y,f0,sp,ap,stft_x,stft_y,w_x,w_y) = encode(x, fs, 5)    

#reldif=np.abs(stft_x-stft_y)/(np.abs(stft_x)+1)
#plot(reldif.sum(axis=0))

#print('{}'.format(reldif.flatten().sum()))

wmean=np.abs(np.log(w_x+.1)-np.log(w_y+.1)).flatten().mean()
mel_x=librosa.feature.melspectrogram(x,fs)
mel_y=librosa.feature.melspectrogram(y,fs)

mel=np.abs(np.log(mel_x)-np.log(mel_y)).flatten().mean()

print('wmean:%g, mel:%g'%(wmean,mel))
#sa.play_buffer(x.astype('int16'),1,2,fs)

#sa.play_buffer(y.astype('int16'),1,2,fs)

#%%


#mel_x=librosa.feature.melspectrogram(x,fs)
#mel_y=librosa.feature.melspectrogram(y,fs)

path="/home/eugening/Neural/MachineLearning/Speech/TrainingData/LJSpeech-1.0/wavs/LJ001-0003.wav"
(fs,x1) = wav.read(path)
x1=x1[0:x.size]
vv=encode(x1,fs,5)