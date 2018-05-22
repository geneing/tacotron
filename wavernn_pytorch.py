#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

#from util import audio

#%%
class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "train.txt"), sep='|', 
                                    header=None, names=['spec','mel','wav','mel_len','wav_len','text'])
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = {}
        wav_path=os.path.join(self.root_dir, self.metadata.wav[idx])
        sample['text'] = self.metadata.text[idx]
        sample['wav'] = np.load(wav_path)
        return sample

root_dir='/home/eugening/Neural/MachineLearning/Speech/TrainingData/LJSpeech-1.0.taco'
ds = SpeechDataset(root_dir)

#%%

quant = 256
hidden_size = 896
eye = np.eye(quant,dtype=float32)
def mergeSamples(inp):
    wavlist=[]
    strlist=[]
    for v in inp:
        quantized = (((v['wav']+1.)/2.)*quant).astype(int)
        
        wavlist.append(quantized)
        strlist.append(v['text'])
    maxlen = max([x.shape[0] for x in wavlist])
    
    wavlist = np.asarray([  np.pad(x, ((0,maxlen-x.shape[0]),), 'constant') for x in wavlist])
    return {'wav':wavlist.T, 'text':strlist}

dl = DataLoader( ds, batch_size = 4, shuffle=True, collate_fn = mergeSamples )

#%%

gru = torch.nn.GRUCell(quant, hidden_size)
O1 = torch.nn.Linear(hidden_size, hidden_size//2)
relu = torch.nn.ReLU()
O2 = torch.nn.Linear(hidden_size//2, quant)


for sample in dl:
    wav = sample['wav']
    h_rnn = torch.zeros([wav.shape[1], hidden_size])
    
    for i in xrange(np.min(1000,wav.shape[0])):
        coded_wav = torch.tensor(eye[ wav[i, :] ])
        h_rnn = gru(coded_wav, h_rnn)
        o1 = relu(O1(h_rnn))
        o2 = O2(o1)    
        
    break


