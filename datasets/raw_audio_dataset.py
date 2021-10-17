import torch
import torchaudio

import glob
from torch.utils.data import Dataset
from utils.signal_processing import get_rnd_audio,extract_label
from pandas import read_csv

class raw_audio_dataset(Dataset):
    
    def __init__(self,wav_dir, annotation_file, input_size, transform=None, target_transform=None):
        """
        Initialises the audio dataset
        """
        self.audio_files        = glob.glob(wav_dir+"/**/*.WAV",recursive=True)
        self.label              = read_csv(annotation_file,sep="\t")
        self.transform          = transform
        self.target_transform   = target_transform
        self.input_size         =input_size
        
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.audio_files)
        
    def __getitem__(self,idx):
        audio_path  = self.audio_files[idx]	# .iloc[idx, 0] to get the correct audio_path
        audio,b,e   = get_rnd_audio(audio_path,self.input_size) # already a tensor
        
        label       = extract_label(self.label.iloc[idx],b,e)

        
        if self.transform:
            audio = self.transform(audio)  # which audio transforms are usually used? 
        if self.target_transform:
            label = self.target_transform(label)
            
        return audio,label
            
       