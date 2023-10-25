import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
#from utils import transforms
from utils_dir import transforms
import torchvision
import torchaudio.transforms as T_audio
import torchaudio
import pandas as pd
import os
import numpy as np
import imageio
import random
import collections
import csv
import librosa

import config

labels_file = pd.read_csv('./data/labeled_ADSMI/labels_int.csv', index_col=0)
class MyDataset_finetune(data.Dataset):
    
    def __init__(self, train=True):
        self.root = './data/labeled_ADSMI/labeled_data_2013-535/'
        self.train = train
        
        #getting name of all files inside the all of the train_folds
        temp = os.listdir(self.root)
        temp.sort()
        self.file_names = []
        self.class_ids = []
        if self.train:
            for i in range(len(labels_file["Label_int"])):
                if int(labels_file["fold"][i]) in config.ADSMI_train_folds:
                    self.file_names.append(labels_file["filename"][i])
                    self.class_ids.append(labels_file["Label_int"][i])
        else:
            for i in range(len(labels_file["Label_int"])):
                if int(labels_file["fold"][i]) in config.ADSMI_test_fold:
                    self.file_names.append(labels_file["filename"][i])
                    self.class_ids.append(labels_file["Label_int"][i])
      
        
        if self.train:
            self.wave_transforms = torchvision.transforms.Compose([  
                                                              transforms.RandomScale(max_scale = 1.25), 
                                                              transforms.RandomPadding(out_len = 320000),
                                                              transforms.RandomCrop(out_len = 320000)])

            
        else: #for test
            self.wave_transforms = torchvision.transforms.Compose([ 
                                                              transforms.RandomPadding(out_len = 320000),
                                                             transforms.RandomCrop(out_len = 320000)])

        self.mel_spectrogram = T_audio.MelSpectrogram(sample_rate=32000, n_fft=1024, hop_length=320,n_mels=128)
        self.amplitude_to_db = T_audio.AmplitudeToDB()

    
    def __len__(self):
        return len(self.file_names)
    
    

    def __getitem__(self, index):
        file_name = self.file_names[index]  
        path = self.root + file_name
        
        # Using torchaudio to load waveform
        wave, rate = torchaudio.load(path)
        wave = wave.squeeze(0)  # Convert to numpy array for processing

        # Identifying the label of the sample from its name
        class_id = int(self.class_ids[index])
        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]
        
        # Normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        nonzero_indices = wave.nonzero()

        if nonzero_indices.size(0) > 0:
            start = nonzero_indices[:, 1].min().item()
            end = nonzero_indices[:, 1].max().item()
            wave = wave[:, start: end + 1]
        else:
            # Create a default or zero tensor of appropriate shape for empty waveforms
            default_spec_shape = (config.channels, default_freq_dim, default_time_dim)
            return file_name, torch.zeros(default_spec_shape), class_id
        

        wave = self.wave_transforms(wave).float().squeeze(0)

        # Using torchaudio to compute melspectrogram
        s = T_audio.MelSpectrogram(sample_rate=32000, n_fft=1024, hop_length=320,n_mels=128)(wave).unsqueeze(0)
        
        # Masking the spectrograms
        log_s = T_audio.FrequencyMasking(freq_mask_param=80)(s)
        log_s = T_audio.TimeMasking(time_mask_param=80)(s)
        log_s = self.amplitude_to_db(log_s.squeeze(0))

        # Creating 3 channels by copying log_s 3 times 

        if config.channels == 3:
            spec = torch.stack([log_s, log_s, log_s]) # Using stack instead of cat

        else:
            spec = log_s
        

        return file_name, spec, class_id
    
def create_generators_finetune():
    train_dataset = MyDataset_finetune(train=True)
    test_dataset = MyDataset_finetune(train=False)
    

    train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=2 ,drop_last=False)
    
    test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=2 ,drop_last=False)
    
    return train_loader, test_loader

