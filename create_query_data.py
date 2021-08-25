import random
import numpy as np
import librosa
import soundfile as sf
import sys
import os
import json
import argparse
import uuid
import warnings
from audiomentations import Compose,Shift,PitchShift,TimeStretch,AddImpulseResponse,FrequencyMask,TimeMask,ClippingDistortion,AddBackgroundNoise,Gain



parser = argparse.ArgumentParser(description='Script for creating Query dataset')
parser.add_argument('--test_dir', default='', type=str, metavar='PATH',
                    help='directory containing test dataset')
parser.add_argument('--noise_dir', default='', type=str, metavar='PATH',
                    help='directory containing noise data')
parser.add_argument('--ir_dir', default='', type=str, metavar='PATH',
                    help='directory containing IR data')
parser.add_argument('--length', nargs='+', type=int,
                    help='length of query')

root = os.path.dirname(__file__)


# ir_dir = os.path.join(root,'data/ir_filters')
# noise_dir = os.path.join(root,'data/Noises_unsampled')
# data_dir = os.path.join(root,'data/test_data/fma_large')
json_path = os.path.join(root,'data/fma_large.json')



# for fname in os.listdir(validation_dir):
#     path = os.path.join(validation_dir, fname)
#     audio, sr = librosa.load(path, sr=8000, mono=True)
#     if len(audio) < 80000:
#         print(fname," : ",len(audio))
args = parser.parse_args()
offset_list = args.length
# print(offset_list)
data_dir = args.test_dir
noise_dir = args.noise_dir
ir_dir = args.ir_dir

# dataset = {}
# idx = 0
# for filename in os.listdir(data_dir):
#   if filename.endswith(".wav") or filename.endswith(".mp3"): 
#     dataset[idx] = filename
#     idx += 1
# with open(json_path, 'w') as fp:
#     json.dump(dataset, fp)
def irconv(ir_dir, x, p):
    if random.random() < p:
        r1 = random.randrange(len(os.listdir(ir_dir)))
        fpath = os.path.join(ir_dir, os.listdir(ir_dir)[r1])
        x_ir, fs = librosa.load(fpath, sr=None)
        fftLength = np.maximum(len(x), len(x_ir))
        X = np.fft.fft(x, n=fftLength)
        X_ir = np.fft.fft(x_ir, n=fftLength)
        x_aug = np.fft.ifft(np.multiply(X_ir, X))[0:len(x)].real
        if np.max(np.abs(x_aug)) == 0:
            pass
        else:
            x_aug = x_aug / np.max(np.abs(x_aug))  # Max-normalize
    
    else: 
        x_aug = x
    
    return x_aug.astype(np.float32)
 
for offset in offset_list:
    SAMPLE_RATE = 8000
    
    validation_dir = os.path.join(root,'data/fma_large_'+str(offset)+'sec_2K')
    # validation_dir = os.path.join(root,'data/eval_test')
        
    
    if not os.path.exists(validation_dir):
      os.makedirs(validation_dir)
    with open(json_path) as f:
        ref = json.load(f)
    iters = 2000
    augment = Compose([
        # Shift(min_fraction=-0.2, max_fraction=0.2, rollover=False),
        # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        # TimeStretch(min_rate=0.8, max_rate=3, p=0.5),
        # AddImpulseResponse(ir_path=ir_dir, p=1),
        # FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=1),
        # TimeMask(min_band_part=0.1, max_band_part=1),
        ClippingDistortion(),
        AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=7,p=1),
        Gain(),
        # Mp3Compression()
        ])
    
    i = 0
    while i < iters:
        r1 = random.randrange(len(ref.keys()))
        fpath = os.path.join(data_dir, ref[str(r1)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(fpath, sr=8000, mono=True)
        except Exception:
            continue
        
    
            
        if i % 50 == 0:
            print(f"Step [{i}/{iters}]")
        offset_frame = int(SAMPLE_RATE*offset)
        if len(audio) < offset_frame:
            continue
        # try:    
        #     r2 = np.random.randint(0,len(audio)-offset_frame)
        # except ValueError:
        #     print("audio length error = ",len(audio)/8000.0)
        #     continue
        r2 = np.random.randint(0,len(audio)-offset_frame)
        
        audioData = audio[r2:r2+offset_frame]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            augmented_samples = augment(samples=audioData, sample_rate=SAMPLE_RATE)
            augmented_samples = irconv(ir_dir, augmented_samples, p=1)
        fname = ref[str(r1)].split(".mp3")[0] + "-" + str(uuid.uuid4()) + "-" + str(r2) +".wav"
        sf.write(os.path.join(validation_dir,fname), augmented_samples, SAMPLE_RATE, format='WAV')
        i+=1
