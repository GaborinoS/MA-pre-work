import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd


#own modules
from Pipelines.Pipeline_FT_SA import MyPipeline
import config

labels_file = pd.read_csv('./data/labeled_ADSMI/labels_int.csv', index_col=0)



class MyDataset_finetune_train(Dataset):
    
    def __init__(self, train_df,test_df,train=True, sample_rate=32000, desired_length_in_seconds=10):
        self.root = './data/labeled_ADSMI/labeled_data_2013-535/'
        
        #getting name of all files inside the all of the train_folds
        temp = os.listdir(self.root)
        temp.sort()
        self.file_names = []
        self.class_ids = []
        self.train = train
        self.sample_rate = config.goal_sr_labeled
        
        
        if self.train:
            self.file_names = train_df["filename"].values
            self.class_ids = train_df["Label_int"].values
        else:
            self.file_names = test_df["filename"].values
            self.class_ids = test_df["Label_int"].values

        
        self.pipeline = MyPipeline(config.goal_sr_labeled, device='cuda', desired_length_in_seconds=desired_length_in_seconds, train=self.train)
        self.pipeline.to(device=torch.device("cuda"), dtype=torch.float32)    
    
    def __len__(self):
        return len(self.file_names)
    


    def __getitem__(self, index):
        file_name = self.file_names[index]  
        path = self.root + file_name
        
        # Using torchaudio to load waveform
        waveform, sample_rate_new = torchaudio.load(path)
        waveform = waveform.to(device=torch.device("cuda"), dtype=torch.float32)

        mel_spec = self.pipeline(sample_rate_new,waveform, goal_r=self.sample_rate)

        class_id = self.class_ids[index]

        return mel_spec, class_id


class MyDataset_finetune_val(Dataset):
    
    def __init__(self, val_df, sample_rate=32000, desired_length_in_seconds=10):
        self.root = './data/labeled_ADSMI/labeled_data_2013-535/'
        
        #getting name of all files inside the all of the train_folds
        temp = os.listdir(self.root)
        temp.sort()
        self.file_names = []
        self.class_ids = []
        self.sample_rate = config.goal_sr_labeled
        
        
        self.file_names = val_df["filename"].values
        self.class_ids = val_df["Label_int"].values

        
        self.pipeline = MyPipeline(config.goal_sr_labeled, device='cuda', desired_length_in_seconds=desired_length_in_seconds, train=False)
        self.pipeline.to(device=torch.device("cuda"), dtype=torch.float32)    
    
    def __len__(self):
        return len(self.file_names)
    


    def __getitem__(self, index):
        file_name = self.file_names[index]  
        path = self.root + file_name
        
        # Using torchaudio to load waveform
        waveform, sample_rate_new = torchaudio.load(path)
        waveform = waveform.to(device=torch.device("cuda"), dtype=torch.float32)

        mel_spec = self.pipeline(sample_rate_new,waveform,goal_r=self.sample_rate)

        class_id = self.class_ids[index]

        return mel_spec, class_id




def create_generators_finetune_train(train_df,test_df):
    train_dataset = MyDataset_finetune_train(train_df,test_df,sample_rate=config.goal_sr_labeled, desired_length_in_seconds=config.desired_length_in_seconds)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)

    test_dataset = MyDataset_finetune_train(train_df,test_df,train=False, desired_length_in_seconds=config.desired_length_in_seconds)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)
    
    return train_loader, test_loader

def create_generators_finetune_val(val_df):
    val_dataset = MyDataset_finetune_val(val_df=val_df, sample_rate=config.goal_sr_labeled,desired_length_in_seconds=config.val_sound_length)
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0 ,drop_last=False)
    
    return val_loader