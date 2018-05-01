#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string

#import matplotlib.pyplot as plt
import time, sys, math
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
#import tensorflow.contrib.eager as tfe
import os
import librosa

from scipy.io import wavfile

#%%

base_dir = '/home/eugening/Neural/MachineLearning/Speech/TrainingData/'
input = 'LJSpeech-1.0.taco/train.txt'
input_path = os.path.join(base_dir, input)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()), write_graph=True, write_grads=True)


hparams.batch_size = 2
hparams.hidden_size = 896
hparams.quantization = 256

coord = tf.train.Coordinator()
with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams, batches_per_group=1)

sess = tf.InteractiveSession()
tf.keras.backend.set_session(sess)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                                                                                                                                                                                                     
feeder.start_in_session(sess)


#model = WaveRNN()

wav_target = feeder.wav_targets
input_val = feeder.mel_targets

#%%

#wav_target_arr=sess.run([wav_target])[0]

inputs = tf.keras.layers.Input(tensor=input_val)
gru = tf.keras.layers.GRU(hparams.hidden_size, return_state = False)
gru_output = gru(inputs)
O1 = tf.keras.layers.Dense(hparams.hidden_size)
O1_output = O1(gru_output)
O2 = tf.keras.layers.Dense(80, activation = "softmax")
wav_output = O2(O1_output)

model = tf.keras.models.Model([inputs], wav_output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'], target_tensors=[input_val])
#model.fit( y=wav_target, batch_size=hparams.batch_size)
model.fit( x=None, y=None, steps_per_epoch=10, batch_size=None, callbacks=[tensorboard])
        