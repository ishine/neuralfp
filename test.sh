#!/bin/sh
python main.py >> results/train.txt
python test.py --test_dir=data/test_data/fma_large --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth
python create_query_data.py --length 1 2 3 5 6 10 --test_dir=data/test_data/fma_large --noise_dir=data/Noises_unsampled --ir_dir=data/ir_filters
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_1sec_2K --eval=True >> results/fma_large_1sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_2sec_2K --eval=True >> results/fma_large_2sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_3sec_2K --eval=True >> results/fma_large_3sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_5sec_2K --eval=True >> results/fma_large_5sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_6sec_2K --eval=True >> results/fma_large_6sec_2K.txt
python test.py --fp_path=fingerprints/fma_large_ver4.pt --model_path=model/model_ver4_epoch_[ENTER EPOCH].pth --query_dir=data/fma_large_10sec_2K --eval=True >> results/fma_large_10sec_2K.txt


