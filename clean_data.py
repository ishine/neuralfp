import os
import librosa
import random
import json
import warnings
import soundfile as sf

root = os.path.dirname(__file__)

data_dir = data_dir = os.path.join(root,"data/test_data/fma_large")
test_dir = os.path.join(root,"data/test_data/10K_subset")
json_path = os.path.join(root,"data/fma_large.json")

if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    
iters = 10000
i = 0
with open(json_path) as f:
    ref = json.load(f)
while i < iters:
     r1 = random.randrange(len(os.listdir(data_dir)))
     fpath = os.path.join(data_dir, ref[str(r1)])
     try:
         with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             audio, sr = librosa.load(fpath, sr=8000, mono=True)
     except Exception:
         continue
    
     dst = os.path.join(test_dir,ref[str(r1)])
     sf.write(dst, audio, sr, format='WAV')
     if i % 50 == 0:
         print(f"Step [{i}/{iters}]")
     i+=1