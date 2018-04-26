#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string

import matplotlib.pyplot as plt
import time, sys, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import librosa

from scipy.io import wavfile

#tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

checkpoint_directory = "/home/eugening/Neural/MachineLearning/Speech/logs/"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")


#%%


def display(string, variables) :
    sys.stdout.write(f'\r{string}' % variables)


def load_wav(filename, sample_rate, encode_16bits=True) :
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode_16bits == True : 
        x = np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)
    return x


def save_wav(y, filename, sample_rate) :
    if y.dtype != 'int16' : y *= 2**15
    y = np.clip(y, -2**15, 2**15 - 1)
    wavfile.write(filename, sample_rate, y.astype(np.int16))


def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15


def time_since(started) :
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60 :
        m = m % 60
        h = int(m // 60)
        return f'{h}h {m}m {s}s'
    else :
        return f'{m}m {s}s'


#%%


sample_rate = 24000


def sine_wave(freq, length, sample_rate=sample_rate) : 
    return np.sin(np.arange(length) * 2 * math.pi * freq / sample_rate).astype(np.float32)


def encode_16bits(x) : 
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def split_signal(x) :
    encoded = encode_16bits(x)
    unsigned = encoded + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


signal = sine_wave(1, 100000)
c, f = split_signal(signal)


#plt.plot(c[30000:32000])
#plt.plot(f[30000:32000])

#%%
class WaveRNN(tf.keras.Model):
    def __init__(self, hidden_size=896, quantisation=256):
        super(WaveRNN, self).__init__()
        
        self.hidden_size = hidden_size

        self.GRUCell = tf.keras.layers.GRUCell(self.hidden_size, use_bias=False)
        
        # Output fc layers
        self.O1 = tf.keras.layers.Dense(self.hidden_size)
        self.O2 = tf.keras.layers.Dense(quantisation)

    def call(self, inputs, training=False) :
        current_input, prev_state = inputs
        hidden, state = self.GRUCell.call(current_input, prev_state)
        # Compute outputs
        out = self.O2(tf.nn.relu(self.O1(hidden)))
        return [out, state]
    
    def init_hidden(self, batch_size=1) :
        return tfe.Variable(tf.zeros([batch_size, self.hidden_size],dtype=tf.float32), name='hidden_val')
        
##%%


with tf.device("/gpu:0"):       
    model = WaveRNN()

##%%
wav = sine_wave(freq=500, length=sample_rate * 30)
wav_classes = np.reshape(wav, (1, -1))


def train(model, optimizer, num_steps, seq_len=960, checkpoint=None) :
    
    start = time.time()
    running_loss = 0
    
    for step in range(num_steps) :
        loss = 0
        hidden = model.init_hidden()
        rand_idx = np.random.randint(0, wav_classes.shape[1] - seq_len - 1)
        
        with tfe.GradientTape() as tape:
            for i in range(seq_len) :
                j = rand_idx + i
                x_input = wav_classes[:, j:j + 1]
                x_input = x_input / 127.5 - 1.
                x_in = tfe.Variable(x_input, dtype=tf.float32, name='x_input')

                out_wav, hidden = model([x_in, hidden])

                loss_curr = tf.losses.sparse_softmax_cross_entropy(y_coarse, out_wav )
                loss += loss_curr

        running_loss += (loss / seq_len)
        
        grad = tape.gradient(loss, model.variables)
        
        print(grad)
        
        optimizer.apply_gradients(zip(grad, model.variables), global_step=tf.train.get_or_create_global_step())
        
        speed = (step + 1) / (time.time() - start)
        
        checkpoint.save(file_prefix=checkpoint_prefix)
        
        sys.stdout.write('\rStep: %i/%i --- NLL: %.2f --- Speed: %.3f batches/second ' % 
                        (step + 1, num_steps, running_loss / (step + 1), speed))

base_dir = '/home/eugening/Neural/MachineLearning/Speech/TrainingData/'
input = 'LJSpeech-1.0.taco/train.txt'
input_path = os.path.join(base_dir, input)

coord = tf.train.Coordinator()
with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope('model') as scope:
    model = WaveRNN(hidden_size=896, quantisation=256)

with tf.device("/gpu:0"):        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    chkpt = tfe.Checkpoint(optimizer=optimizer, model=model)
    
    train(model, optimizer, num_steps=1, checkpoint=chkpt)

#%%

output, c, f = model.generate(5000)

plt.plot(output[:300])

#%%
base_dir = '/home/eugening/Neural/MachineLearning/Speech/TrainingData/'
input = 'LJSpeech-1.0.taco/train.txt'
input_path = os.path.join(base_dir, input)

hparams.batch_size = 8

coord = tf.train.Coordinator()
with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams, batches_per_group=4)

sess = tf.Session()
feeder.start_in_session(sess)

model = WaveRNN()

wav_target = feeder.wav_targets
input_val = feeder.mel_targets

output = []
i=0
while not coord.should_stop():
    
    for 
    o = sess.run(wav_target)
    output.append(o)
    print(i)
    i+=1
    
    coord.request_stop()


        