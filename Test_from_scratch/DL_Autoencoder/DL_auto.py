import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np

#own modules
from Pipelines.Pipeline_FT_SA import MyPipelinePreTrain_auto
import config

root = 'G:/Dokumente/MAData/Unlabeled_all/'
#root = './data/unlabeled_test/'
#root = 'G:/Dokumente/MAData/Unlabeled_klein/'
#root = './data/ESC50/ESC-50-master/audio/'

filenames = os.listdir(root)


n = len(filenames)
f = len(config.ADSMI_train_folds) + 1 # replace with your desired value of f

# Compute quotient and remainder
q, r = divmod(n, f)

# Create the list
FOLDS = []
for i in range(1, f + 1):
    FOLDS.extend([i] * q)

# Add the remainder elements
for i in range(1, r + 1):
    FOLDS.append(i)
#shuffle
random.shuffle(FOLDS)
random.shuffle(FOLDS)
random.shuffle(FOLDS)

#create array 
filenames_fold = np.zeros((len(filenames),2), dtype=object)
filenames_fold[:,0] = filenames
filenames_fold[:,1] = FOLDS

class MyDataset_pretrain(Dataset):
    
    def __init__(self, data_names, path ,train=True, desired_length_in_seconds=10):
        self.root = path
        
        #getting name of all files inside the all of the train_folds
        temp = os.listdir(self.root)
        self.file_names = []
        self.train = train
        self.sample_rate = config.goal_sr_unlabeled
        
        
        if self.train:
            self.file_names = [x for x,y in data_names[:,:] if y in config.ADSMI_train_folds]
            
        else:
            self.file_names = [x for x,y in data_names[:,:] if y in config.ADSMI_test_fold]

        print('Number of files: ', len(self.file_names))

        true_value = self.train or config.val_masked
        print("True Value: ", true_value)
        
        self.pipeline = MyPipelinePreTrain_auto(input_sample_rate=config.goal_sr_labeled, device="cuda", desired_length_in_seconds=desired_length_in_seconds, train=true_value)
        self.pipeline.to(device=torch.device("cuda"), dtype=torch.float32)    

    
    def __len__(self):
        return len(self.file_names)
    


    def __getitem__(self, index):
        file_name_pos = self.file_names[index]  
        path = self.root + file_name_pos


        # Using torchaudio to load waveform
        waveform_pos, sample_rate_new_pos = torchaudio.load(path)
        waveform_pos = waveform_pos.to(device=torch.device("cuda"), dtype=torch.float32)
        

        mel_spec_masked, mel_spec = self.pipeline(sample_rate_new_pos,config.desired_length_in_seconds,waveform_pos,goal_r=config.goal_sr_unlabeled)
        
        #print("Mel Spectrogram Size in __getitem__:", mel_spec.size())  # Debugging statement
        return mel_spec_masked, mel_spec





def create_generators():
    train_dataset = MyDataset_pretrain(data_names=filenames_fold,path=root, desired_length_in_seconds=config.desired_length_in_seconds)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)

    test_dataset = MyDataset_pretrain(data_names=filenames_fold,path=root, desired_length_in_seconds=config.desired_length_in_seconds,train=False)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)
    
    return train_loader, test_loader