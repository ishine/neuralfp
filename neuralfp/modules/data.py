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
    def __init__(self, path, json_dir, transform=None, validate=False):
        # self.dataset = dataset
        self.path = path
        self.transform = transform
        self.validate = validate
        with open(json_dir) as json_file:
            self.filenames  = json.load(json_file)
        # self.input_shape = input_shape
        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.path + "/" + self.filenames[str(idx)]
        try:
            audio, sr = torchaudio.load(datapath)
        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            return self[idx+1]

        audioData = audio.mean(dim=0)
        print("audio length: ",len(audioData))
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audioData = resampler(audioData)    # Downsampling
        spec_func = MelSpectrogram(sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256, n_fft=2048)    

        offset_frame = int(SAMPLE_RATE*offset)
        
        if len(audioData) <= offset_frame:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        
        if not self.validate:
            r = np.random.randint(0,len(audioData)-offset_frame)
            audioData = audioData[r:r+offset_frame]
            audioData_i, audioData_j = self.transform(audioData.numpy())
            # print(audioData.shape,audioData_i.shape,audioData_j.shape)
            audioData_i = torch.from_numpy(audioData_i)
            audioData_j = torch.from_numpy(audioData_j)
    
            specData_i = spec_func(audioData_i)
            specData_i = torchaudio.transforms.AmplitudeToDB()(specData_i)
            specData_i = F.pad(specData_i, (target_len - specData_i.size(-1), 0))
    
            specData_j = spec_func(audioData_j)
            specData_j = torchaudio.transforms.AmplitudeToDB()(specData_j)
            specData_j = F.pad(specData_j, (target_len - specData_j.size(-1), 0))
            return torch.unsqueeze(specData_i, 0), torch.unsqueeze(specData_j, 0)
        
        else:
            chunks = torch.split(audioData, offset_frame)
            spec = []
            if chunks[-1].size(0) < 1024:
                chunks = chunks[:-1]
            for data in chunks:
                  specData = spec_func(data)
                  specData = torchaudio.transforms.AmplitudeToDB()(specData)
                  specData = F.pad(specData, (target_len - specData.size(-1), 0))
                  specData = torch.unsqueeze(specData, 0)
                  spec.append(specData)
            return torch.unsqueeze(torch.cat(spec),1), self.filenames[str(idx)]
    
    def __len__(self):
        return len(self.filenames)
            
    