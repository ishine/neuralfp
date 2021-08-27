#!/bin/sh
python test.py --fp_path=fingerprints/fma_large_ver5.pt --model_path=model/model_ver5_epoch_230.pth --query_dir=data/fma_large_2sec_2K --eval=True >> results/fma_large_2sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver5.pt --model_path=model/model_ver5_epoch_230.pth --query_dir=data/fma_large_3sec_2K --eval=True >> results/fma_large_3sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver5.pt --model_path=model/model_ver5_epoch_230.pth --query_dir=data/fma_large_5sec_2K --eval=True >> results/fma_large_5sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver5.pt --model_path=model/model_ver5_epoch_230.pth --query_dir=data/fma_large_6sec_2K --eval=True >> results/fma_large_6sec_2K.txt


