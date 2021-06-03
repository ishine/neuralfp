import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
import numpy as np
# from pydub import AudioSegment

offset = 1.25
SAMPLE_RATE = 8000

class NeuralfpDataset(Dataset):
    def __init__(self, path, transform):
        # self.dataset = dataset
        self.path = path
        self.transform = transform
        # self.input_shape = input_shape
        self.ignore_idx = []
        self.mixer = torchaudio.transforms.DownmixMono()
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        path = self.path
        audio, sr = torchaudio.load(path)
        audioData = self.mixer(audio[0])
        audioData = audioData[::(int)(sr/SAMPLE_RATE)]     # Downsampling
        r = np.random.randint(0,len(audioData))
        offset_time = sr*offset
        audioData = audioData[r:r+offset_time]               # 1.2 second samples
        
        # if audio.shape[1] < self.input_shape[1]:
        #     self.ignore_idx.append(idx)
        #     return self[idx + 1]
        
        if self.transform:
            audioData = self.transform(samples=audioData.numpy(),sample_rate=SAMPLE_RATE)
            audioData = torch.from_numpy(audioData)
        return audioData
    
    def __len__(self):
        return len(self.dataset)
            
    