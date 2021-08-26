#!/bin/sh
python main.py >> results/train_ver5.txt
python clean_data.py
rm -rf data/fma_large_*



