#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from util import audio

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
import pdb
def mergeSamples(inp):
    wavlist=[]
    strlist=[]
    for v in inp:
        wavlist.append(v['wav'])
        strlist.append(v['text'])
    maxlen = max([len(x) for x in wavlist])
    wavlist = np.asarray([  np.pad(x, ((0,maxlen-x.size),), 'constant') for x in wavlist])
    
    return {'wav':wavlist.T, 'text':strlist}

dl = DataLoader( ds, batch_size = 3, shuffle=False, collate_fn = mergeSamples )
for v in dl: break