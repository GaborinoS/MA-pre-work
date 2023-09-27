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


######US8k DATASET########
class AudioDataset(data.Dataset):

    def __init__(self, train=True):
        
        self.root = './data/US8K/audio/'
        self.train = train
        self.file_paths = [] #only includes the name of the fold and name of the file, like: 'fold2/4201-3-0-0.wav'
        
        if train:
            for f in config.train_folds:
                file_names = os.listdir(self.root + 'fold' + str(f) + '/' )
                
                for name in file_names:
                    if name.split('.')[-1] == 'wav':
                        self.file_paths.append('fold' + str(f) + '/' + name)
        else:
            file_names = os.listdir(self.root + 'fold' + str(config.test_fold[0]) + '/' )
            for name in file_names:
                if name.split('.')[-1] == 'wav':
                    self.file_paths.append('fold' + str(config.test_fold[0]) + '/' + name)
            
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
        return len(self.file_paths)
    
    

    def __getitem__(self, index):
        file_path = self.file_paths[index ]  
        path_pos = self.root + file_path
        wave, rate = librosa.load(path_pos, sr=44100)
        
        #getting negative sample (randomly index of file_path withou index of positive sample)
        file_path_neg = random.choice(self.file_paths)
        while file_path_neg == file_path:
            file_path_neg = random.choice(self.file_paths)
     
        path_neg = self.root + file_path_neg
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

        # Padding or croping
        wave_pos1 = self.wave_transforms(wave)
        wave_pos1.squeeze_(0)

        wave_pos2 = self.wave_transforms(wave)
        wave_pos2.squeeze_(0)

        wave_neg1 = self.wave_transforms(wave_neg)
        wave_neg1.squeeze_(0)

        # Spectrogram augmentation (only for training)
        s_pos1 = librosa.feature.melspectrogram(y=wave_pos1.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_pos1 = librosa.power_to_db(s_pos1, ref=np.max)
        log_s_pos_aug1 = self.spec_transforms(log_s_pos1)

        s_pos2 = librosa.feature.melspectrogram(y=wave_pos2.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_pos2 = librosa.power_to_db(s_pos2, ref=np.max)
        log_s_pos_aug2 = self.spec_transforms(log_s_pos2)

        s_neg1 = librosa.feature.melspectrogram(y=wave_neg1.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512)
        log_s_neg1 = librosa.power_to_db(s_neg1, ref=np.max)
        log_s_neg_aug1 = self.spec_transforms(log_s_neg1)


        #creating 3 channels by copying log_s1 3 times 
        #log_s_pos_aug1 = torch.cat((log_s_pos_aug1, log_s_pos_aug1, log_s_pos_aug1), dim=0)
        #log_s_pos_aug2 = torch.cat((log_s_pos_aug2, log_s_pos_aug2, log_s_pos_aug2), dim=0)
        #log_s_neg_aug1 = torch.cat((log_s_neg_aug1, log_s_neg_aug1, log_s_neg_aug1), dim=0)

        
        return log_s_pos_aug1, log_s_pos_aug2 , log_s_neg_aug1


def create_generators():
    # Assuming the class name is AudioDataset based on previous interactions
    train_dataset = AudioDataset(train=True)
    test_dataset = AudioDataset(train=False)
    
    # Create the data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    return train_loader, test_loader


print('US8k dataset is ready to be used!')


