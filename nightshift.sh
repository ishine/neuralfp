#!/bin/sh
python test.py --test_dir=data/test_data/fma_large --model_path=model/model_ver3_epoch_220.pth
# python create_query_data.py --length 1 2 3 5 6 10 --test_dir=data/test_data/fma_large --noise_dir=data/Noises_unsampled --ir_dir=data/ir_filters
python create_query_data.py --length 10 --test_dir=data/test_data/fma_large --noise_dir=data/Noises_unsampled --ir_dir=data/ir_filters
# python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_260.pth --query_dir=data/fma_large_1sec_2K --eval=True >> results/fma_large_1sec_2K.txt
# python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_260.pth --query_dir=data/fma_large_2sec_2K --eval=True >> results/fma_large_2sec_2K.txt
# python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_260.pth --query_dir=data/fma_large_3sec_2K --eval=True >> results/fma_large_3sec_2K.txt
# python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_260.pth --query_dir=data/fma_large_5sec_2K --eval=True >> results/fma_large_5sec_2K.txt
# python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_260.pth --query_dir=data/fma_large_6sec_2K --eval=True >> results/fma_large_6sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver3.pt --model_path=model/model_ver3_epoch_220.pth --query_dir=data/fma_large_10sec_2K --eval=True
# python main.py >> results/train_ver4.txt

