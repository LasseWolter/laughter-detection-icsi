import sys, numpy as np
sys.path.append('./utils/')
import models, audio_utils
from functools import partial

# takes a batch tuple (X,y)
def add_channel_dim(X):
    return np.expand_dims(X,1)

CONFIG_MAP = {}


CONFIG_MAP['resnet_base'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100) 
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

CONFIG_MAP['resnet_with_augmentation'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}