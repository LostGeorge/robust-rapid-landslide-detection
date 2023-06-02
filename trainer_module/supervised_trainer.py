import pytorch_lightning as pl
import torch
from typing import Any, List, Union, Callable
from torchmetrics import AUROC, AveragePrecision, F1Score

from data_module.single_dm import SingleBeforeAfterCubeDataModule
from trainer_module.helpers import BinarySegmentationMetricsWrapper

class plSupervisedTrainerModule(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss: Union[str, Callable] = 'ce',
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model

        if loss == 'ce' or loss == torch.nn.CrossEntropyLoss:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        self.metrics = BinarySegmentationMetricsWrapper()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        y = y.long()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.metrics.train_auc.update(preds, targets)
        self.metrics.train_auprc.update(preds., targets.)
        self.metrics.train_f1.update(preds., targets.)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.metrics.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.metrics.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.metrics.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "inputs": inputs}

    def on_training_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # log val metrics
        self.metrics.val_auc.update(preds, targets)
        self.metrics.val_auprc.update(preds., targets.)
        self.metrics.val_f1.update(preds., targets.)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.metrics.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.metrics.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.metrics.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        self.metrics.test_auc.update(preds, targets)
        self.metrics.test_auprc.update(preds., targets.)
        self.metrics.test_f1.update(preds., targets.)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auc", self.metrics.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.metrics.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.metrics.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}
    

if __name__ == '__main__':
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(encoder_name='resnet50', in_channels=4, classes=2)
    dm = SingleBeforeAfterCubeDataModule(
        ds_path='data/hokkaido_japan.zarr',
        ba_vars=['vv', 'vh'],
        aggregation='mean',
        sat_orbit_state='d',
        timestep_length=2,
        event_start_date='20180905',
        event_end_date='20180907',
        input_vars=['vv_before', 'vv_after', 'vh_before', 'vh_after'],
        target='landslides',
        include_negatives=False,
        split_fp='data/hokkaido_70_20_10.yaml',
        batch_size=64,
        num_workers=4
    )
    trainer_module = plSupervisedTrainerModule(model=model)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')
    
