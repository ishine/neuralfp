#!/bin/sh
# python create_query_data.py --length 10 --test_dir=data/test_data/fma_large --noise_dir=data/Noises_unsampled --ir_dir=data/IR_unsampled --ps=1
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ps_1 --eval=True >> results/ver6_ps_1.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ps_2 --eval=True >> results/ver6_ps_2.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ps_-1 --eval=True >> results/ver6_ps_-1.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ps_-2 --eval=True >> results/ver6_ps_-2.txt
python test.py --fp_path=fingerprints/fma_large_ver6.pt --model_path=model/model_ver6_epoch_280.pth --query_dir=data/fma_large_10sec_ps_-3 --eval=True >> results/ver6_ps_-3.txt




