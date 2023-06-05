from torchmetrics import AUROC, AveragePrecision, F1Score, Precision, Recall, JaccardIndex
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def domain_confusion_loss(disc_logits):
    '''
    disc_logits: (N, D) tensor where D is the number of domains
    '''
    return -torch.mean(torch.sum(F.log_softmax(disc_logits, dim=1), dim=1) / disc_logits.shape[1])

class BinarySegmentationMetricsWrapper(pl.LightningModule):
    
    def __init__(self, device=None) -> None:
        super().__init__()
        splits = ['train_', 'val_', 'test_']
        self.metrics_dict = nn.ModuleDict({split: self._generate_metrics_group(device) for split in splits})
    
    def _generate_metrics_group(self, device=None):
        if device is None:
            return nn.ModuleDict({
                'auprc': AveragePrecision(task='binary'),
                # 'f1': F1Score(task='binary'),
                'precision': Precision(task='binary'),
                'recall': Recall(task='binary'),
                'jaccard': JaccardIndex(task='binary'),
            })
        else:
            return nn.ModuleDict({
                'auprc': AveragePrecision(task='binary').to(device),
                # 'f1': F1Score(task='binary').to(device),
                'precision': Precision(task='binary').to(device),
                'recall': Recall(task='binary').to(device),
                'jaccard': JaccardIndex(task='binary').to(device),
            })

class FocalTverskyLoss(nn.Module):

    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 1

    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky


