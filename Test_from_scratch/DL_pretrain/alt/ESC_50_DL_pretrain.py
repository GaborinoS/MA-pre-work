
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
#from utils import transforms
from utils_dir import transforms
import torchvision

import os
import numpy as np
import imageio
import random
import collections
import csv
import librosa

import config

######ESC50 DATASET########
class AudioDataset(data.Dataset):

    def __init__(self, train=True):
            self.root = './data/ESC50/ESC-50-master/audio/'
            self.train = train
            
            #getting name of all files inside the all of the train_folds
            temp = os.listdir(self.root)
            temp.sort()
            self.file_names = []
            if train:
                for i in range(len(temp)):
                    if int(temp[i].split('-')[0]) in config.train_folds:
                        self.file_names.append(temp[i])
            else:
                for i in range(len(temp)):
                    if int(temp[i].split('-')[0]) in config.test_fold:
                        self.file_names.append(temp[i])
            
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
        file_name_pos = self.file_names[index ]  
        path_pos = self.root + file_name_pos
        wave, rate = librosa.load(path_pos, sr=44100)
        
        #getting negative sample (randomly index of file_names withou index of positive sample)
        file_name_neg = self.file_names[random.randint(0, len(self.file_names) - 1)]
        while file_name_neg == file_name_pos:
            file_name_neg = self.file_names[random.randint(0, len(self.file_names) - 1)]
        
        path_neg = self.root + file_name_neg
        wave_neg, rate_neg = librosa.load(path_neg, sr=44100)

        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]
        if wave_neg.ndim == 1:
            wave_neg = wave_neg[:, np.newaxis]
        
    # normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        if np.abs(wave_neg.max()) > 1.0:
            wave_neg = transforms.scale(wave_neg, wave_neg.min(), wave_neg.max(), -1.0, 1.0)
        wave_neg = wave_neg.T * 32768.0

        # Remove silent sections
        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]  

        start = wave_neg.nonzero()[1].min()
        end = wave_neg.nonzero()[1].max()
        wave_neg = wave_neg[:, start: end + 1]

        


        wave_copy_pos1 = np.copy(wave)
        wave_copy_pos1 = self.wave_transforms(wave_copy_pos1)
        wave_copy_pos1.squeeze_(0)
        
        s = librosa.feature.melspectrogram(y=wave_copy_pos1.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_pos1 = librosa.power_to_db(s, ref=np.max)

        wave_copy_pos2 = np.copy(wave)
        wave_copy_pos2 = self.wave_transforms(wave_copy_pos2)
        wave_copy_pos2.squeeze_(0)

        s = librosa.feature.melspectrogram(y=wave_copy_pos2.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_pos2 = librosa.power_to_db(s, ref=np.max)


        wave_copy_neg = np.copy(wave_neg)
        wave_copy_neg = self.wave_transforms(wave_copy_neg)
        wave_copy_neg.squeeze_(0)

        s = librosa.feature.melspectrogram(y=wave_copy_neg.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_neg = librosa.power_to_db(s, ref=np.max)


        
        # masking the spectrograms
        log_s_po_aug1 = self.spec_transforms(log_s_pos1)
        log_s_po_aug2 = self.spec_transforms(log_s_pos2)
        log_s_neg_aug1 = self.spec_transforms(log_s_neg)
        
        #print(config.channels)

        if config.channels == 3:
            #creating 3 channels by copying log_s1 3 times 
            log_s_po_aug1 = torch.cat((log_s_po_aug1, log_s_po_aug1, log_s_po_aug1), dim=0)
            log_s_po_aug2 = torch.cat((log_s_po_aug2, log_s_po_aug2, log_s_po_aug2), dim=0)
            log_s_neg_aug1 = torch.cat((log_s_neg_aug1, log_s_neg_aug1, log_s_neg_aug1), dim=0)
        
        return log_s_po_aug1, log_s_po_aug2 , log_s_neg_aug1


def create_generators():
    # Assuming the class name is AudioDataset based on previous interactions
    train_dataset = AudioDataset(train=True)
    test_dataset = AudioDataset(train=False)
    
    # Create the data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    return train_loader, test_loader