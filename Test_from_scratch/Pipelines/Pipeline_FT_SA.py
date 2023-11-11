
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import torchvision.models as models
import torchaudio
import os

import random
import torch.nn as nn
import numpy as np

import config

class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate,
        device,
        n_fft=2048,
        hop_length = 512,
        n_mels=128,  
        win_length = 2048,
        desired_length_in_seconds=5,
        train=True,
    ):
        super().__init__()
        
        self.train = train
        self.desired_length_in_seconds = desired_length_in_seconds
        self.sample_rate = input_sample_rate
        self.device = device  

        #Functions  
        self.mel_spectrogram = T.MelSpectrogram(
        sample_rate=input_sample_rate,  
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        win_length=win_length,
        window_fn=torch.hann_window  
        ).to(self.device)

        self.amplitude = T.AmplitudeToDB().to(self.device)
        self.time_stretch = T.TimeStretch().to(self.device)

        
        self.additive_noise = T.Vol(1.1).to(self.device)  # Increase volume by 10% (for additive noise)
        self.volume_perturbation = T.Vol(0.9).to(self.device)  # Decrease volume by 10%

        self.spec_aug = torch.nn.Sequential( 
            T.FrequencyMasking(freq_mask_param=10).to(self.device),#10
            T.TimeMasking(time_mask_param=90).to(self.device),#90
            #T.TimeStretch( random.uniform(0.95,1.05),fixed_rate=True).to(self.device)
        )

    #@staticmethod
    def random_crop_or_pad(self, waveform: torch.Tensor, sample_rate, desired_length_in_seconds=5) -> torch.Tensor:
            """
            Randomly crops the waveform to the desired length in seconds.
            If the waveform is shorter than the desired length, it will be padded with zeros.
            """
            desired_length = desired_length_in_seconds * sample_rate
            current_length = waveform.shape[1]

            # If the waveform is shorter than desired, pad it with zeros
            side = random.randint(0,2)

            if current_length < desired_length:
                if side == 0:
                    padding_needed = desired_length - current_length
                    left_pad = padding_needed // 2
                    right_pad = padding_needed - left_pad
                    waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
                elif side == 1:
                    padding_needed = desired_length - current_length
                    left_pad = padding_needed
                    right_pad = 0
                    waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
                else:
                    padding_needed = desired_length - current_length
                    left_pad = 0
                    right_pad = padding_needed
                    waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
            
            # Calculate the starting point for cropping
            start_idx = random.randint(0, waveform.shape[1] - desired_length)
            return waveform[:, start_idx:start_idx+desired_length]

    def forward(self,sr,length_in_sec, waveform: torch.Tensor,goal_r=32000) -> torch.Tensor:
        
        waveform = waveform.to(self.device)
        waveform = self.random_crop_or_pad(waveform, sr, length_in_sec)
        # Apply pitch shift

        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)
    
        # Convert to power spectrogram
        spec = self.mel_spectrogram(waveform)
        
        # Apply SpecAugment
        if self.train:
            spec = self.spec_aug(spec)

        # Convert to decibel
        spec = self.amplitude(spec).squeeze(0)
        #spec = spec - spec.max()
        #spec = self.fix_spectrogram_length(spec, target_length=624)

        if config.channels == 3:
            spec = torch.stack([spec, spec, spec]) 

        

        return spec
    



class MyPipelinePreTrain(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate,
        device,
        n_fft=2048,
        hop_length = 512,
        n_mels=128,  
        win_length = 2048,
        desired_length_in_seconds=5,
        train=True,
    ):
        super().__init__()
        
        self.train = train
        self.desired_length_in_seconds = desired_length_in_seconds
        self.sample_rate = input_sample_rate
        self.device = device  

        #Functions  
        self.mel_spectrogram = T.MelSpectrogram(
        sample_rate=input_sample_rate,  
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        win_length=win_length,
        window_fn=torch.hann_window  
        ).to(self.device)

        self.amplitude = T.AmplitudeToDB().to(self.device)
        self.time_stretch = T.TimeStretch().to(self.device)

        
        self.additive_noise = T.Vol(1.1).to(self.device)  # Increase volume by 10% (for additive noise)
        self.volume_perturbation = T.Vol(0.9).to(self.device)  # Decrease volume by 10%

        self.spec_aug = torch.nn.Sequential( 
            T.FrequencyMasking(freq_mask_param=25).to(self.device),#10
            T.TimeMasking(time_mask_param=90).to(self.device),#90
            #T.TimeStretch( random.uniform(0.95,1.05),fixed_rate=True).to(self.device)
        )

    def random_crop_or_pad(self, waveform: torch.Tensor, sample_rate, desired_length_in_seconds=5) -> torch.Tensor:
        """
        Randomly crops the waveform to the desired length in seconds.
        If the waveform is shorter than the desired length, it will be padded with zeros.
        """
        desired_length = desired_length_in_seconds * sample_rate
        current_length = waveform.shape[1]

        # If the waveform is shorter than desired, pad it with zeros
        side = random.randint(0,2)

        if current_length < desired_length:
            if side == 0:
                padding_needed = desired_length - current_length
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
            elif side == 1:
                padding_needed = desired_length - current_length
                left_pad = padding_needed
                right_pad = 0
                waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
            else:
                padding_needed = desired_length - current_length
                left_pad = 0
                right_pad = padding_needed
                waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
        
        # Calculate the starting point for cropping
        start_idx = random.randint(0, waveform.shape[1] - desired_length)
        return waveform[:, start_idx:start_idx+desired_length]
    
    def forward(self,sr, waveform: torch.Tensor,goal_r=50000) -> torch.Tensor:
        

        waveform = waveform.to(self.device)
        waveform = self.random_crop_or_pad(waveform, sr, self.desired_length_in_seconds)
        # Apply pitch shift

        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)
    
        
        #t=0
        if self.train:
            # Additive white noise
            if random.random() < 0.5:
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.004)
                waveform = waveform + self.additive_noise(noise)

            # Volume perturbation
            if random.random() < 0.5:
                waveform = self.volume_perturbation(waveform)
            '''
            if random.random() < 1:
                ps = T.PitchShift(sample_rate=sr,n_steps=0.1).to(self.device)
                waveform = ps(waveform)

            #Time stretch without changing pitch
            if random.random() < 0.5 and t!=1:
                speed_factor = random.uniform(0.9, 1.1)
                resampler = T.Resample(
                orig_freq=self.sample_rate, new_freq=int(sr * speed_factor)
                ).to(self.device)
                waveform = resampler(waveform)
            '''

        
        # Convert to power spectrogram
        spec = self.mel_spectrogram(waveform)
        
        # Apply SpecAugment
        if self.train:
            spec = self.spec_aug(spec)

        # Convert to decibel
        spec = self.amplitude(spec).squeeze(0)
        #spec = spec - spec.max()

        if config.channels == 3:
            spec = torch.stack([spec, spec, spec]) 

        

        return spec

