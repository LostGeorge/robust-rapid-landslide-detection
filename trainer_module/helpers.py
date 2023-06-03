from torchmetrics import AUROC, AveragePrecision, F1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def domain_confusion_loss(disc_logits):
    '''
    disc_logits: (N, D) tensor where D is the number of domains
    '''
    return -torch.mean(torch.sum(F.log_softmax(disc_logits), dim=1) / disc_logits.shape[1])

class BinarySegmentationMetricsWrapper(pl.LightningModule):
    
    def __init__(self, device=None) -> None:
        super().__init__()
        if device is None:
            self.train_auc = AUROC(task='binary', pos_label=1)
            self.train_f1 = F1Score(task='binary')
            self.train_auprc = AveragePrecision(task='binary', pos_label=1)
            self.val_auc = AUROC(task='binary', pos_label=1)
            self.val_f1 = F1Score(task='binary')
            self.val_auprc = AveragePrecision(task='binary', pos_label=1)
            self.test_auc = AUROC(task='binary', pos_label=1)
            self.test_auprc = AveragePrecision(task='binary', pos_label=1)
            self.test_f1 = F1Score(task='binary')
        else:
            self.train_auc = AUROC(task='binary', pos_label=1).to(device)
            self.train_f1 = F1Score(task='binary').to(device)
            self.train_auprc = AveragePrecision(task='binary', pos_label=1).to(device)
            self.val_auc = AUROC(task='binary', pos_label=1).to(device)
            self.val_f1 = F1Score(task='binary').to(device)
            self.val_auprc = AveragePrecision(task='binary', pos_label=1).to(device)
            self.test_auc = AUROC(task='binary', pos_label=1).to(device)
            self.test_auprc = AveragePrecision(task='binary', pos_label=1).to(device)
            self.test_f1 = F1Score(task='binary').to(device)


