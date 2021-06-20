import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram

offset = 1.25
SAMPLE_RATE = 8000
target_len = 60

class NeuralfpDataset(Dataset):
    def __init__(self, path, json_dir, transform):
        # self.dataset = dataset
        self.path = path
        self.transform = transform
        with open(json_dir) as json_file:
            self.filenames  = json.load(json_file)
        # self.input_shape = input_shape
        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.path + "/" + self.filenames[str(idx)]
        audio, sr = torchaudio.load(datapath)
        audioData = audio.mean(dim=0)
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audioData = resampler(audioData)    # Downsampling
        
        offset_frame = int(SAMPLE_RATE*offset)
        
        if len(audioData) <= len(offset_frame):
            self.ignore_idx.append(idx)
            return self[idx + 1]
        
        offset_frame = int(SAMPLE_RATE*offset)
        r = np.random.randint(0,len(audioData)-offset_frame)
        audioData = audioData[r:r+offset_frame]
                       
        
        audioData_i, audioData_j = self.transform(audioData.numpy())
        # print(audioData.shape,audioData_i.shape,audioData_j.shape)
        audioData_i = torch.from_numpy(audioData_i)
        audioData_j = torch.from_numpy(audioData_j)

        spec_func = MelSpectrogram(sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256, n_fft=2048)    
        specData_i = spec_func(audioData_i)
        specData_i = torchaudio.transforms.AmplitudeToDB()(specData_i)
        specData_i = F.pad(specData_i, (target_len - specData_i.size(-1), 0))

        specData_j = spec_func(audioData_j)
        specData_j = torchaudio.transforms.AmplitudeToDB()(specData_j)
        specData_j = F.pad(specData_j, (target_len - specData_j.size(-1), 0))
        return torch.unsqueeze(specData_i, 0), torch.unsqueeze(specData_j, 0)
    
    def __len__(self):
        return len(self.filenames)
            
    