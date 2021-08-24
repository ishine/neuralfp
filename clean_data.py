import os
import numpy as np
import librosa
import json
import warnings
from scipy.io.wavfile import write
import torchaudio

root = os.path.dirname(__file__)

data_dir = os.path.join(root,"data/test_data/fma_large")
test_dir = os.path.join(root,"data/test_data/3K_subset")
json_path = os.path.join(root,"data/fma_large.json")



    


dataset = {}
idx = 0
for filename in os.listdir(data_dir)[:50000]:
  print(filename)
  if filename.endswith(".wav") or filename.endswith(".mp3"): 
    dataset[idx] = filename
    idx += 1
with open(json_path, 'w') as fp:
    json.dump(dataset, fp)


