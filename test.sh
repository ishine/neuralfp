#!/bin/sh
# python create_query_data.py --length 10 --test_dir=data/test_data/fma_large --noise_dir=data/Noises_unsampled --ir_dir=data/IR_unsampled --ts=1.1
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_1.1 --eval=True >> results/ver6_ts_1.1.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_1.2 --eval=True >> results/ver6_ts_1.2.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_1.3 --eval=True >> results/ver6_ts_1.3.txt
python test.py --fp_path=fingerprints/fma_large_ver.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_0.9 --eval=True >> results/ver6_ts_0.9.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_0.8 --eval=True >> results/ver6_ts_0.8.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ts_0.7 --eval=True >> results/ver6_ts_0.7.txt





