import torch
import torchaudio

import glob
from torch.utils.data import Dataset
from pamai.utils.signal_processing import get_rnd_audio,extract_label_bat
from pandas import read_csv
from os import path
class raw_audio_dataset(Dataset):
    
    def __init__(self,wav_dir, annotation_file, input_size, transform=None, target_transform=None):
        """
        Initialises the audio dataset
        """
        self.audio_files        = read_csv(annotation_file)['File name']
        self.label              = read_csv(annotation_file)
        self.transform          = transform
        self.target_transform   = target_transform
        self.input_size         = input_size
        self.wav_dir            = wav_dir
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.audio_files)
        
    def __getitem__(self,idx):
        audio_path  = path.join(self.wav_dir,self.audio_files[idx])	# .iloc[idx, 0] to get the correct audio_path
        audio,b,e   = get_rnd_audio(audio_path,self.input_size) # already a tensor
        label       = extract_label_bat(self.label.iloc[idx],b,e)

        
        if self.transform:
            audio = self.transform(audio)  # which audio transforms are usually used? 
        if self.target_transform:
            label = self.target_transform(label)
            
        return audio,label
            
       