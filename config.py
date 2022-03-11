import sys, numpy as np
sys.path.append('./utils/')
import models
from functools import partial
from pathlib import Path

MODEL_MAP = {}

MODEL_MAP['resnet_base'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100) 
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet_with_augmentation'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}

FEAT = {
    "num_samples": 100,
    "num_filters": 44
}

root_path = Path(__file__).absolute().parent
ANALYSIS= {
    "transcript_dir": str(root_path / 'data/icsi/transcripts'),
    "speech_dir": str(root_path / 'data/icsi/speech'),
    "plots_dir": 'plots',

    # Indices are loaded from disk if possible. This option forces re-computation 
    # If True analyse.py will take a lot longer
    "force_index_recompute": False 
}

ANALYSIS['model'] = {
    # Min-length used for parsing the transcripts
    "min_length": 0.2,

    # Frame duration used for parsing the transcripts
    "frame_duration": 1  # in ms
}

ANALYSIS['train'] = {
    # How long each sample for training should be 
    "subsample_duration": 1.0,  # in s
    "random_seed": 23,

    # Used in creation of train, val and test df in 'create_data_df'
    "float_decimals": 2,  # number of decimals to round floats to
    # Test uses the remaining fraction
    "train_val_test_split": [0.8, 0.1],
}