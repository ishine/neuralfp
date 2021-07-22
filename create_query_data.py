import random
import numpy as np
import librosa
import soundfile as sf
import sys
import os
import json
import uuid
import warnings
from audiomentations import Compose,Shift,PitchShift,TimeStretch,AddImpulseResponse,FrequencyMask,TimeMask,ClippingDistortion,AddBackgroundNoise,Gain


root = os.path.dirname(__file__)

validation_dir = os.path.join(root,'data/gtzan_2sec_2K')
ir_dir = os.path.join(root,'data/ir_filters')
noise_dir = os.path.join(root,'data/noise')
data_dir = os.path.join(root,'data/test_data/fma_large')
json_path = os.path.join(root,'data/fma_large.json')


SAMPLE_RATE = 8000
offset = 2

# ref = {}
# idx = 0
# for filename in os.listdir(data_dir):
#   if filename.endswith(".wav") or filename.endswith(".mp3"): 
#     ref[idx] = filename
#     idx += 1

if not os.path.exists(validation_dir):
  os.makedirs(validation_dir)
with open(json_path) as f:
    ref = json.load(f)
iters = 2000
augment = Compose([
    # Shift(min_fraction=-0.2, max_fraction=0.2, rollover=False),
    # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    # TimeStretch(min_rate=0.8, max_rate=3, p=0.5),
    AddImpulseResponse(ir_path=ir_dir, p=1),
    # FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=1),
    # TimeMask(min_band_part=0.1, max_band_part=1),
    ClippingDistortion(),
    AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=1),
    Gain(),
    # Mp3Compression()
    ])

i = 0
while i < iters:
    r1 = random.randrange(len(os.listdir(data_dir)))
    fpath = os.path.join(data_dir, ref[str(r1)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(fpath, sr=8000, mono=True)
    except Exception:
        iters+=1
        
    if i % 50 == 0:
        print(f"Step [{i}/{iters}]")
    offset_frame = int(SAMPLE_RATE*offset)
    r2 = np.random.randint(0,len(audio)-offset_frame)
    audioData = audio[r2:r2+offset_frame]
    augmented_samples = augment(samples=audioData, sample_rate=SAMPLE_RATE)
    fname = ref[str(r1)].split(".mp3")[0] + "-" + str(uuid.uuid4()) + ".wav"
    sf.write(os.path.join(validation_dir,fname), augmented_samples, SAMPLE_RATE, format='WAV')
    i+=1
