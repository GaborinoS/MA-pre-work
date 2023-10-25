import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
#from utils import transforms
from utils_dir import transforms
import torchvision

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
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(), 
                                                              transforms.RandomScale(max_scale = 1.25), 
                                                              transforms.RandomPadding(out_len = 220500),
                                                              transforms.RandomCrop(out_len = 220500)])
             
            
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() , 
									transforms.FrequencyMask(max_width = config.freq_masks_width, numbers = config.freq_masks), 
									transforms.TimeMask(max_width = config.time_masks_width, numbers = config.time_masks)])
            
        else: #for test
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(),
                                                              transforms.RandomPadding(out_len = 220500),
                                                             transforms.RandomCrop(out_len = 220500)])
        
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() ])

    
    def __len__(self):
        return len(self.file_names)
    
    

    def __getitem__(self, index):
        file_name = self.file_names[index ]  
        path = self.root + file_name
        wave, rate = librosa.load(path, sr=44100) 
        
        #identifying the label of the sample from its name
        class_id = int(self.class_ids[index])
        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]
		
	# normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        nonzero_rows, nonzero_cols = wave.nonzero()

        if nonzero_rows.size > 0:
            start = nonzero_cols.min()
            end = nonzero_cols.max()
            wave = wave[:, start: end + 1]
        else:
            # Create a default or zero tensor of appropriate shape for empty waveforms
            default_spec_shape = (config.channels, default_freq_dim, default_time_dim)  # You have to define default_freq_dim and default_time_dim based on what's appropriate for your application
            return file_name, torch.zeros(default_spec_shape), class_id
        wave = wave[:, start: end + 1]  
        
        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)
        
        s = librosa.feature.melspectrogram(y=wave_copy.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512) 
        log_s = librosa.power_to_db(s, ref=np.max)
        
	# masking the spectrograms
        log_s = self.spec_transforms(log_s)
        
        
        #creating 3 channels by copying log_s1 3 times 
        if config.channels == 3:
            spec = torch.cat((log_s, log_s, log_s), dim=0)
        else:
            spec = log_s
        
        
        return file_name, spec, class_id
        


def create_generators_finetune():
    train_dataset = MyDataset_finetune(train=True)
    test_dataset = MyDataset_finetune(train=False)
    

    train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)
    
    test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)
    
    return train_loader, test_loader