import torch
import torch.nn as nn
import torch.nn.functional as F


def domain_confusion_loss(disc_logits):
    '''
    disc_logits: (N, D) tensor where D is the number of domains
    '''
    return -torch.sum(F.log_softmax(disc_logits), dim=1) / disc_logits.shape[1]



