#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time, sys, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from scipy.io import wavfile

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

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
    def __init__(self, hidden_size=896, quantisation=256) :
        super(WaveRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.split_size = hidden_size // 2
        
        # The main hidden state matmul
        #self.hidden_size,
        self.R = tf.keras.layers.Dense(3 * self.hidden_size, use_bias=False)
        
        # Output fc layers
        self.O1 = tf.keras.layers.Dense(self.split_size)
        self.O2 = tf.keras.layers.Dense(quantisation)
        self.O3 = tf.keras.layers.Dense(self.split_size)
        self.O4 = tf.keras.layers.Dense(quantisation)
        
        # Input fc layers
        self.I_coarse = tf.keras.layers.Dense( 3 * self.split_size, use_bias=False)
        self.I_fine = tf.keras.layers.Dense( 3 * self.split_size, use_bias=False)

        # biases for the gates
        self.bias_u = tfe.Variable(tf.zeros(self.hidden_size), name='bias_u')
        self.bias_r = tfe.Variable(tf.zeros(self.hidden_size), name='bias_r')
        self.bias_e = tfe.Variable(tf.zeros(self.hidden_size), name='bias_e')

    def call(self, inputs, training=False) :
        prev_y, prev_hidden, current_coarse = inputs
        # Main matmul - the projection is split 3 ways
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e = tf.split(R_hidden, 3, axis=1) #tf.keras.layers.Lambda(lambda z: tf.split(z, 3, axis=1), arguments=R_hidden)
        
        # Project the prev input 
        coarse_input_proj = self.I_coarse(prev_y)
        I_coarse_u, I_coarse_r, I_coarse_e = \
            tf.split(coarse_input_proj, 3, axis=1)
            #tf.keras.layers.Lambda(lambda z: tf.split(z, 3, axis=1), arguments=coarse_input_proj)
        
        # Project the prev input and current coarse sample
        fine_input = tf.concat([prev_y, current_coarse], axis=1)
        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = \
            tf.split(fine_input_proj, 3, axis=1)
        
        # concatenate for the gates
        # TODO: Simplify all of this business 
        I_u = tf.concat([I_coarse_u, I_fine_u], axis=1)
        I_r = tf.concat([I_coarse_r, I_fine_r], axis=1)
        I_e = tf.concat([I_coarse_e, I_fine_e], axis=1)
        
        # Compute all gates for coarse and fine 
        u = tf.sigmoid(R_u + I_u + self.bias_u, name='u')
        r = tf.sigmoid(R_r + I_r + self.bias_r, name='r')
        e = tf.tanh(r * R_e + I_e + self.bias_e, name='e')
        hidden = u * prev_hidden + (1. - u) * e
        
        # Split the hidden state
        hidden_coarse, hidden_fine = tf.split(hidden, 2, axis=1)
        
        # Compute outputs 
        out_coarse = self.O2(tf.nn.relu(self.O1(hidden_coarse)))
        out_fine = self.O4(tf.nn.relu(self.O3(hidden_fine)))

        return [out_coarse, out_fine, hidden]
    
        
    def generate(self, seq_len) :
        
        # First split up the biases for the gates 
        b_coarse_u, b_fine_u = torch.split(self.bias_u, self.split_size)
        b_coarse_r, b_fine_r = torch.split(self.bias_r, self.split_size)
        b_coarse_e, b_fine_e = torch.split(self.bias_e, self.split_size)
        
        # Lists for the two output seqs
        c_outputs, f_outputs = [], []
        
        # Some initial inputs
        out_coarse = Variable(torch.LongTensor([0])).cuda()
        out_fine = Variable(torch.LongTensor([0])).cuda()
        
        # We'll meed a hidden state
        hidden = self.init_hidden()
        
        # Need a clock for display
        start = time.time()
        
        # Loop for generation
        for i in range(seq_len) :
            
            # Split into two hidden states
            hidden_coarse, hidden_fine = \
                torch.split(hidden, self.split_size, dim=1)
            
            # Scale and concat previous predictions
            out_coarse = out_coarse.unsqueeze(0).float() / 127.5 - 1.
            out_fine = out_fine.unsqueeze(0).float() / 127.5 - 1.
            prev_outputs = torch.cat([out_coarse, out_fine], dim=1)
            
            # Project input 
            coarse_input_proj = self.I_coarse(prev_outputs)
            I_coarse_u, I_coarse_r, I_coarse_e = \
                torch.split(coarse_input_proj, self.split_size, dim=1)
            
            # Project hidden state and split 6 ways
            R_hidden = self.R(hidden)
            R_coarse_u , R_fine_u, \
            R_coarse_r, R_fine_r, \
            R_coarse_e, R_fine_e = torch.split(R_hidden, self.split_size, dim=1)
        
            # Compute the coarse gates
            u = F.sigmoid(R_coarse_u + I_coarse_u + b_coarse_u)
            r = F.sigmoid(R_coarse_r + I_coarse_r + b_coarse_r)
            e = F.tanh(r * R_coarse_e + I_coarse_e + b_coarse_e)
            hidden_coarse = u * hidden_coarse + (1. - u) * e
            
            # Compute the coarse output
            out_coarse = self.O2(F.relu(self.O1(hidden_coarse)))
            posterior = F.softmax(out_coarse, dim=1).view(-1)
            distrib = torch.distributions.Categorical(posterior)
            out_coarse = distrib.sample()
            c_outputs.append(out_coarse)
            
            # Project the [prev outputs and predicted coarse sample]
            coarse_pred = out_coarse.float() / 127.5 - 1.
            fine_input = torch.cat([prev_outputs, coarse_pred.unsqueeze(0)], dim=1)
            fine_input_proj = self.I_fine(fine_input)
            I_fine_u, I_fine_r, I_fine_e = \
                torch.split(fine_input_proj, self.split_size, dim=1)
            
            # Compute the fine gates
            u = F.sigmoid(R_fine_u + I_fine_u + b_fine_u)
            r = F.sigmoid(R_fine_r + I_fine_r + b_fine_r)
            e = F.tanh(r * R_fine_e + I_fine_e + b_fine_e)
            hidden_fine = u * hidden_fine + (1. - u) * e
        
            # Compute the fine output
            out_fine = self.O4(F.relu(self.O3(hidden_fine)))
            posterior = F.softmax(out_fine, dim=1).view(-1)
            distrib = torch.distributions.Categorical(posterior)
            out_fine = distrib.sample()
            f_outputs.append(out_fine)
    
            # Put the hidden state back together
            hidden = torch.cat([hidden_coarse, hidden_fine], dim=1)
            
            # Display progress
            speed = (i + 1) / (time.time() - start)
            display('Gen: %i/%i -- Speed: %i',  (i + 1, seq_len, speed))
        
        coarse = torch.stack(c_outputs).squeeze(1).cpu().data.numpy()
        fine = torch.stack(f_outputs).squeeze(1).cpu().data.numpy()        
        output = combine_signal(coarse, fine)
        
        return output, coarse, fine
        
             
    def init_hidden(self, batch_size=1) :
        return tfe.Variable(tf.zeros([batch_size, self.hidden_size],dtype=float32), name='hidden_val')
        
    
    
    def print_stats(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)


##%%

model = WaveRNN()

##%%
x = sine_wave(freq=500, length=sample_rate * 30)
coarse_classes, fine_classes = split_signal(x)
coarse_classes = np.reshape(coarse_classes, (1, -1))
fine_classes = np.reshape(fine_classes, (1, -1))

def train(model, optimizer, num_steps, seq_len=960) :
    
    start = time.time()
    running_loss = 0
    
    for step in range(num_steps) :
        
        loss = 0
        hidden = model.init_hidden()
        #optimizer.zero_grad()
        rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
        
        with tfe.GradientTape() as tape:
            for i in range(seq_len) :

                j = rand_idx + i

                x_coarse = coarse_classes[:, j:j + 1]
                x_fine = fine_classes[:, j:j + 1]
                x_input = np.concatenate([x_coarse, x_fine], axis=1)
                x_input = x_input / 127.5 - 1.
                x_input = tfe.Variable(x_input, dtype=float32)

                y_coarse = coarse_classes[:, j + 1]
                y_fine = fine_classes[:, j + 1]
                y_coarse = tfe.Variable(y_coarse)
                y_fine = tfe.Variable(y_fine)

                current_coarse = tf.cast(y_coarse, dtype=float32) / 127.5 - 1.
                current_coarse = tf.expand_dims(current_coarse, axis=0)

                out_coarse, out_fine, hidden = model([x_input, hidden, current_coarse])

                loss_coarse = tf.losses.sparse_softmax_cross_entropy(y_coarse, out_coarse )
                loss_fine = tf.losses.sparse_softmax_cross_entropy(y_fine, out_fine)
                loss += (loss_coarse + loss_fine)

        running_loss += (loss / seq_len)
        
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grad, model.variables), global_step=tf.train.get_or_create_global_step())
        
        speed = (step + 1) / (time.time() - start)
        
        sys.stdout.write('\rStep: %i/%i --- NLL: %.2f --- Speed: %.3f batches/second ' % 
                        (step + 1, num_steps, running_loss / (step + 1), speed))    

with tf.device("/gpu:0"):        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train(model, optimizer, num_steps=50)

#%%

output, c, f = model.generate(5000)

plt.plot(output[:300])
        