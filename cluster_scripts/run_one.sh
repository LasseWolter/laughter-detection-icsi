#!/bin/bash
#conda init bash
conda activate pt
cd ~/git/laughter-detection
python train.py --config=resnet_base --checkpoint_dir=./test_check --data_root=./data/icsi/
