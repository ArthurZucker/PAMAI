import torch
import torchaudio

import glob
from torch.utils.data import Dataset

from pandas import read_csv

class raw_audio_dataset(Dataset):
    
    def __init__(self,wav_dir, annotation_file, transform=None, target_transform=None):
        """
        Initialises the audio dataset
        """
        self.audio_files        = glob.glob(wav_dir+"/**/*.WAV",recursive=True)
        self.label              = read_csv(annotation_file,sep="\t")
        self.transform          = transform
        self.target_transform   = target_transform
    
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.audio_files)
        
    def __getitem__(self,idx):
        audio_path  = self.audio_files[idx]	# .iloc[idx, 0] to get the correct audio_path
        audio       = torchaudio.load(audio_path) # already a tensor
        label       = self.label.iloc[idx, 1] 
        
        if self.transform:
            audio = self.transform(audio)  # which audio transforms are usually used? 
        if self.target_transform:
            label = self.target_transform(label)
            
        return audio,label
            
    
    