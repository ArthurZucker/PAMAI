import torch
import torchaudio

import os
from torch.utils.data import Dataset

from pandas import read_csv

def audio_dataset(Dataset):
    
    def __init__(self, file_path, annotation_path, transform=None, target_transform=None):
        self.audio_files = os.listdir(file_path)
        self.label = read_csv(annotation_path)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self,idx):
        audio_path = self.audio_files[idx]	# .iloc[idx, 0] to get the correct audio_path
        audio = torchaudio.load(audio_path)
        label = self.label.iloc[idx, 1] 
    
    