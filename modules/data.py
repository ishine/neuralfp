import json
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram

offset = 1.25
SAMPLE_RATE = 8000

class NeuralfpDataset(Dataset):
    def __init__(self, path, json_dir, transform):
        # self.dataset = dataset
        self.path = path
        self.transform = transform
        with open(json_dir) as json_file:
            self.filenames  = json.load(json_file)
        # self.input_shape = input_shape
        self.ignore_idx = []
        # self.mixer = torchaudio.transforms.DownmixMono()
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.path + "/" + self.filenames[str(idx)]
        audio, sr = torchaudio.load(datapath)
        audioData = torch.mean(audio, dim=0, keepdim=True)
        audioData = audioData[::(int)(sr/SAMPLE_RATE)]     # Downsampling
        r = np.random.randint(0,len(audioData))
        offset_frame = sr*offset
        audioData = audioData[r:r+offset_frame]               # 1.25 second samples
        
        # if audio.shape[1] < self.input_shape[1]:
        #     self.ignore_idx.append(idx)
        #     return self[idx + 1]
        
        if self.transform:
            audioData = self.transform(samples=audioData.numpy(),sample_rate=SAMPLE_RATE)
            audioData = torch.from_numpy(audioData)
            
        specData = MelSpectrogram(sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256, n_fft=512)
        return specData
    
    def __len__(self):
        return len(self.filenames)
            
    