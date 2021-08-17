import os
import numpy as np
import librosa
import json
import warnings
from scipy.io.wavfile import write
import torchaudio

root = os.path.dirname(__file__)

data_dir = data_dir = os.path.join(root,"data/test_data/fma_large")
test_dir = os.path.join(root,"data/test_data/3K_subset")
json_path = os.path.join(root,"data/fma_large.json")

if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    
iters = 3000
i = 0
with open(json_path) as f:
    ref = json.load(f)
while i < iters:
    fpath = os.path.join(data_dir, ref[str(i)])
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         audio, sr = librosa.load(fpath)
    # except Exception:
    #     i+=1
    #     iters+=1
    #     continue
    audio, sr = torchaudio.load(fpath)
    dst = os.path.join(test_dir,ref[str(i)])
    write(dst, audio, sr, audio.astype(np.int16))
    if i % 50 == 0:
        print(f"Step [{i}/{iters}]")
    i+=1



# del_list = []
# flist = os.listdir(data_dir)
# for i,fname in enumerate(flist):
#     fpath = os.path.join(data_dir, fname)
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             audio, sr = torchaudio.load(fpath)
#     except Exception:
#         os.remove(fpath)
#         del_list.append(fname)
#     if i % 50 == 0:
#         print(f"Step [{i}/{len(os.listdir(data_dir))}]")
# print(del_list)

