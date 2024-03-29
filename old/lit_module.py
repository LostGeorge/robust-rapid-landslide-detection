'''
Code from Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes
by Boehm et. al 2022, NIPS workshop on Tackling Climate Change with Machine Learning.
https://github.com/iprapas/landslide-sar-unet
'''

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from typing import Any, List
from torchmetrics import AUROC, AveragePrecision, F1Score

class plMLP(pl.LightningModule):
    def __init__(
        self,
        encoder,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        num_channels: int = 1,
        loss='mse'
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.encoder = encoder
        self.prediction_head = torch.nn.Linear(in_features=2048*7*7, out_features=2, bias=True)

        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)[-1] # grab the final encoded layer
        out = self.prediction_head(encoded.flatten())
        return out

    def step(self, batch: Any):
        x, y = batch
        y = y.long()
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "inputs": inputs}

    def on_training_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # log val metrics

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}


class plUNET(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            num_channels: int = 1,
            loss='ce'
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = smp.UnetPlusPlus(encoder_name='resnet50', in_channels=num_channels, classes=2)

        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.train_auc = AUROC(task='binary', pos_label=1)
        self.train_f1 = F1Score(task='binary')
        self.train_auprc = AveragePrecision(task='binary', pos_label=1)
        self.val_auc = AUROC(task='binary', pos_label=1)
        self.val_f1 = F1Score(task='binary')
        self.val_auprc = AveragePrecision(task='binary', pos_label=1)
        self.test_auc = AUROC(task='binary', pos_label=1)
        self.test_auprc = AveragePrecision(task='binary', pos_label=1)
        self.test_f1 = F1Score(task='binary')


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        y = y.long()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.train_auc.update(preds, targets)
        self.train_auprc.update(preds.flatten(), targets.flatten())
        self.train_f1.update(preds.flatten(), targets.flatten())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "inputs": inputs}

    def on_training_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # log val metrics
        self.val_auc.update(preds, targets)
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.val_f1.update(preds.flatten(), targets.flatten())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        self.test_auc.update(preds, targets)
        self.test_auprc.update(preds.flatten(), targets.flatten())
        self.test_f1.update(preds.flatten(), targets.flatten())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}
