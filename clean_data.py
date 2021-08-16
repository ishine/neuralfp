import os
import librosa
import json
import warnings
import soundfile as sf

root = os.path.dirname(__file__)

data_dir = data_dir = os.path.join(root,"data/test_data/fma_large")
test_dir = os.path.join(root,"data/test_data/10K_subset")
json_path = os.path.join(root,"data/fma_large.json")

# if not os.path.exists(test_dir):
#     os.mkdir(test_dir)
    
# iters = 10000
# i = 0
# with open(json_path) as f:
#     ref = json.load(f)
# while i < iters:
#      fpath = os.path.join(data_dir, ref[str(i)])
#      try:
#          with warnings.catch_warnings():
#              warnings.simplefilter("ignore")
#              audio, sr = librosa.load(fpath, sr=8000, mono=True)
#      except Exception:
#          i+=1
#          iters+=1
#          continue
    
#      dst = os.path.join(test_dir,ref[str(i)])
#      sf.write(dst, audio, sr, format='WAV')
#      if i % 50 == 0:
#          print(f"Step [{i}/{iters}]")
#      i+=1
del_list = []
flist = os.listdir(data_dir)
for i,fname in enumerate(flist):
    fpath = os.path.join(data_dir, fname)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(fpath, sr=8000, mono=True)
    except Exception:
        os.remove(fpath)
        del_list.append(fname)
    
    print(f"Step [{i}/{len(os.listdir(data_dir))}]")
print(del_list)

