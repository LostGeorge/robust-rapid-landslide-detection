import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import random
import yaml

"""
Convenience Functions
"""

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed, workers=True)

def parse_yaml(path) -> dict:
    with open(path, 'r') as yml:
        return yaml.safe_load(yml)
    
def get_encoder_output_channels(encoder_name):
    if encoder_name == 'resnet18' or encoder_name == 'resnet34':
        # return 512
        return 1024
    elif encoder_name == 'resnet50' or encoder_name == 'renet101' or encoder_name == 'resnet152':
        # return 2048
        return 3904
    
"""
Evaluation
"""


