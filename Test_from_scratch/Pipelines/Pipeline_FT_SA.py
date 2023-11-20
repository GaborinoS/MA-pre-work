
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
            T.FrequencyMasking(freq_mask_param=config.freq_mask_param).to(self.device),#10
            T.TimeMasking(time_mask_param=config.time_mask_param).to(self.device),#90
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
            waveform = waveform[:, start_idx:start_idx+desired_length]
            #print("Post crop/pad waveform size:", waveform.size())
            return waveform
    


    def forward(self,sr,length_in_sec, waveform: torch.Tensor,goal_r=32000) -> torch.Tensor:
        
        waveform = waveform.to(self.device)
        
        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)

        waveform = self.random_crop_or_pad(waveform, goal_r, length_in_sec)
        #print("Pre Mel-Spec waveform size:", waveform.size())

    
        # Convert to power spectrogram
        spec = self.mel_spectrogram(waveform)
        #print("Post Mel-Spec size:", spec.size())
        # Apply SpecAugment
        if self.train:
            spec = self.spec_aug(spec)

        # Convert to decibel
        spec = self.amplitude(spec).squeeze(0)
        #spec = spec - spec.max()


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
            T.FrequencyMasking(freq_mask_param=config.freq_mask_param).to(self.device),#10
            T.TimeMasking(time_mask_param=config.time_mask_param).to(self.device),#90
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
        
        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)
        waveform = self.random_crop_or_pad(waveform, goal_r, self.desired_length_in_seconds)
        # Apply pitch shift


    
        
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

class MyPipelinePreTrain_2(torch.nn.Module):
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

        
        self.additive_noise = T.Vol(random.uniform(1,1.1)).to(self.device)  # Increase volume by 10% (for additive noise)
        self.volume_perturbation = T.Vol(random.uniform(0.9,1)).to(self.device)  # Decrease volume by 10%

        self.spec_aug = torch.nn.Sequential( 
            T.FrequencyMasking(freq_mask_param=config.freq_mask_param).to(self.device),#10
            T.TimeMasking(time_mask_param=config.time_mask_param).to(self.device),#90
            #T.TimeStretch( random.uniform(0.95,1.05),fixed_rate=True).to(self.device)
        )

    def random_crop_or_pad(self, waveform: torch.Tensor, sample_rate, desired_length_in_seconds=5) -> (torch.Tensor, torch.Tensor):
        desired_length = desired_length_in_seconds * sample_rate
        current_length = waveform.shape[1]
        waveform2 = waveform.clone()

        # Padding if current length is shorter than desired
        if current_length < desired_length:
            padding_needed = desired_length - current_length
            w1 = self.pad_waveform(waveform, padding_needed)
            w2 = self.pad_waveform(waveform2, padding_needed)
        else:
            # Cropping if current length is longer than desired
            max_shift_samples = int(sample_rate * config.max_sec_shift)  # Maximum shift of 0.4 seconds in samples

            start_idx_1 = random.randint(0, max(0, current_length - desired_length))
            start_idx_2 = min(max(0, start_idx_1 + random.randint(-max_shift_samples, max_shift_samples)), current_length - desired_length)

            w1 = waveform[:, start_idx_1:start_idx_1 + desired_length]
            w2 = waveform2[:, start_idx_2:start_idx_2 + desired_length]

        return w1, w2

    def pad_waveform(self, waveform, padding_needed):
        left_pad = random.randint(0, padding_needed)
        right_pad = padding_needed - left_pad
        return torch.nn.functional.pad(waveform, (left_pad, right_pad))

    
    def forward(self,sr, waveform: torch.Tensor,goal_r=50000) -> torch.Tensor:
        
        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)
        waveform, waveform2 = self.random_crop_or_pad(waveform, goal_r, self.desired_length_in_seconds)
        
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
            if random.random() < 0.5 :
                speed_factor = random.uniform(0.9, 1.1)
                resampler = T.Resample(
                orig_freq=self.sample_rate, new_freq=int(sr * speed_factor)
                ).to(self.device)
                waveform = resampler(waveform)
            '''
        if self.train:
            # Additive white noise
            if random.random() < 0.5:
                noise = torch.randn_like(waveform2) * random.uniform(0.001, 0.004)
                waveform2 = waveform2 + self.additive_noise(noise)

            # Volume perturbation
            if random.random() < 0.5:
                waveform2 = self.volume_perturbation(waveform2)
            
            '''
            if random.random() < 1:
                ps = T.PitchShift(sample_rate=sr,n_steps=0.1).to(self.device)
                waveform2 = ps(waveform2)
            
            #Time stretch without changing pitch
            if random.random() < 0.5 :
                speed_factor = random.uniform(0.9, 1.1)
                resampler = T.Resample(
                orig_freq=self.sample_rate, new_freq=int(sr * speed_factor)
                ).to(self.device)
                waveform2 = resampler(waveform2)
            '''
        
        # Convert to power spectrogram
        spec = self.mel_spectrogram(waveform)
        spec2 = self.mel_spectrogram(waveform2)
        
        # Apply SpecAugment
        if self.train:
            spec = self.spec_aug(spec)
            spec2 = self.spec_aug(spec2)

        # Convert to decibel
        spec = self.amplitude(spec).squeeze(0)
        spec2 = self.amplitude(spec2).squeeze(0)
        #spec = spec - spec.max()

        if config.channels == 3:
            spec = torch.stack([spec, spec, spec]) 
            spec2 = torch.stack([spec2, spec2, spec2])

        

        return spec, spec2
    



class MyPipelinePreTrain_auto(torch.nn.Module):
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

        
        self.additive_noise = T.Vol(random.uniform(1,1.1)).to(self.device)  # Increase volume by 10% (for additive noise)
        self.volume_perturbation = T.Vol(random.uniform(0.9,1)).to(self.device)  # Decrease volume by 10%

        self.spec_aug = torch.nn.Sequential( 
            T.FrequencyMasking(freq_mask_param=config.freq_mask_param).to(self.device),#10
            T.TimeMasking(time_mask_param=config.time_mask_param).to(self.device),#90
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


    

    def forward(self,sr,desired_length_in_seconds, waveform: torch.Tensor,goal_r=50000) -> torch.Tensor:
        
        if sr != goal_r:
            resampler = T.Resample(orig_freq=sr, new_freq=goal_r).to(self.device)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)
        waveform = self.random_crop_or_pad(waveform, goal_r,desired_length_in_seconds)
        waveform2 = waveform.clone()
        #t=0

        # Convert to power spectrogram
        spec = self.mel_spectrogram(waveform)
        spec2 = self.mel_spectrogram(waveform2)
        
        # Apply SpecAugment
        if self.train:
            spec = self.spec_aug(spec)
            #spec2 = self.spec_aug(spec2)

        # Convert to decibel
        spec = self.amplitude(spec).squeeze(0)
        spec2 = self.amplitude(spec2).squeeze(0)
        #spec = spec - spec.max()

        if config.channels == 3:
            spec = torch.stack([spec, spec, spec]) 
            spec2 = torch.stack([spec2, spec2, spec2])

        

        return spec, spec2