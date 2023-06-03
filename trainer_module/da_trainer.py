import pytorch_lightning as pl
import torch
from typing import Any, List, Union, Callable, Optional, Dict, Tuple
from torchmetrics import AUROC, AveragePrecision, F1Score

from trainer_module.helpers import BinarySegmentationMetricsWrapper, domain_confusion_loss

class plDATrainerModule(pl.LightningModule):
    def __init__(
            self,
            encoder: torch.nn.Module,
            models: List[torch.nn.Module],
            discriminator: torch.nn.Module,
            model_losses: List[Union[str, Callable]],
            lr: float,
            model_lambdas: List[float] = [1, 1, 1],
            disc_lambda: float = 1,
            da_lambda: float = 1,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.models = models
        self.discriminator = discriminator

        self.model_losses = []
        for loss in model_losses:
            if loss == 'ce' or loss == torch.nn.CrossEntropyLoss:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                raise NotImplementedError
            self.model_losses.append(criterion)
        self.disc_loss = torch.nn.CrossEntropyLoss()
        self.da_loss = domain_confusion_loss
        
        metrics = []
        for _ in range(len(models)):
            metrics.append(BinarySegmentationMetricsWrapper(device))
        self.metrics = torch.nn.ModuleList(metrics)


    def forward(self, inputs: List[torch.Tensor]):
        enc_out_stages = [self.encoder(x) for x in inputs] # List[List[tensor]]
        disc_out = self.discriminator(torch.cat([stages[-1] for stages in enc_out_stages], dim=0))
        dec_out = [self.models[i].decoder(stages) for i, stages in enumerate(enc_out_stages)]
        seg_out = [self.models[i].segmentation_head(out)[:, 0, ...] for i, out in enumerate(dec_out)]
        return seg_out, disc_out

    def training_step(self, batch_dict: Dict[int, Tuple], batch_idx: int):
        '''
        Note: We assume batch_dict is idx -> input, label in the input order
        '''
        seg_out, disc_out = self([batch_dict[i][0] for i in range(len(batch_dict))])
        
        seg_losses = [loss_fn(seg_out[i], batch_dict[i][1]) for i, loss_fn in enumerate(self.model_losses)]
        seg_loss = torch.sum(torch.stack(
            [seg_losses[i] * self.hparams.model_lambdas[i] for i in range(len(seg_losses))]))
        disc_labels = torch.cat([torch.ones(len(batch_dict[i][0]) * i) for i in range(len(batch_dict))])
        disc_loss = self.disc_loss(disc_out, disc_labels) * self.hparams.disc_lambda
        da_loss = self.da_loss(disc_out) * self.hparams.da_lambda
        
        self.log("train/seg_loss_total", seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/da_loss", da_loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, metrics_obj in enumerate(self.metrics):
            metrics_obj.train_auc.update(seg_out[i], batch_dict[i][1])
            metrics_obj.train_auprc.update(seg_out[i], batch_dict[i][1])
            metrics_obj.train_f1.update(seg_out[i], batch_dict[i][1])

            self.log(f"train/{i}/seg_loss", seg_losses[i], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/{i}/auc", metrics_obj.train_auc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/{i}/auprc", metrics_obj.train_auprc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/{i}/f1", metrics_obj.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        seg_optimizer, disc_optimizer, da_optimizer = self.optimizers()
        seg_optimizer.zero_grad()
        self.manual_backward(seg_loss)
        seg_optimizer.step()
        disc_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        disc_optimizer.step()
        da_optimizer.zero_grad()
        self.manual_backward(da_loss)
        da_optimizer.step()

        return None # since manual optimization

    def on_training_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        x, targets = batch
        preds = self.models[dataloader_idx](x)[:, 0, ...]

        self.metrics[dataloader_idx].val_auc.update(preds, targets)
        self.metrics[dataloader_idx].val_auprc.update(preds, targets)
        self.metrics[dataloader_idx].val_f1.update(preds, targets)

        self.log(f"val/{dataloader_idx}/auc", self.metrics[dataloader_idx].val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{dataloader_idx}/auprc", self.metrics[dataloader_idx].val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{dataloader_idx}/f1", self.metrics[dataloader_idx].val_f1, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        x, targets = batch
        preds = self.models[dataloader_idx](x)[:, 0, ...]

        self.metrics[dataloader_idx].test_auc.update(preds, targets)
        self.metrics[dataloader_idx].test_auprc.update(preds.flatten(), targets.flatten())
        self.metrics[dataloader_idx].test_f1.update(preds.flatten(), targets.flatten())

        self.log(f"test/{dataloader_idx}/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"test/{dataloader_idx}/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"test/{dataloader_idx}/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)


    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        seg_params = list(self.encoder.parameters())
        for model in self.models:
            seg_params.extend(list(model.decoder.parameters()))
            seg_params.extend(list(model.segmentation_head.parameters()))
        
        seg_optimizer = torch.optim.Adam(params=seg_params, lr=self.hparams.lr)
        disc_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.hparams.lr)
        da_optimizer = torch.optim.Adam(params=self.encoder.parameters(), lr=self.hparams.lr)
        # TODO: maybe add schedulers (not sure if good idea)
        return seg_optimizer, disc_optimizer, da_optimizer


if __name__ == '__main__':
    import segmentation_models_pytorch as smp
    from models.da_models import instantiate_da_models
    from models.discriminator import MLPDiscriminator
    from data_module.multi_dm import MultiBeforeAfterCubeDataModule

    dm = MultiBeforeAfterCubeDataModule([
        {
            'ds_path': 'data/hokkaido_japan.zarr',
            'ba_vars': ['vv', 'vh'],
            'aggregation': 'mean',
            'sat_orbit_state': 'd',
            'timestep_length': 1,
            'event_start_date': '20180905',
            'event_end_date': '20180907',
            'input_vars': ['vv_before', 'vv_after', 'vh_before', 'vh_after'],
            'target': 'landslides',
            'include_negatives': False,
            'split_fp': 'data/hokkaido_70_20_10.yaml',
            'batch_size': 8,
            'num_workers': 2
        },
        {
            'ds_path': 'data/kaikoura_newzealand.zarr',
            'ba_vars': ['vv', 'vh'],
            'aggregation': 'mean',
            'sat_orbit_state': 'ascending',
            'timestep_length': 1,
            'event_start_date': '20161114',
            'event_end_date': '20161115',
            'input_vars': ['vv_before', 'vv_after', 'vh_before', 'vh_after'],
            'target': 'landslides',
            'include_negatives': False,
            'split_fp': 'data/kaikoura_70_20_10.yaml',
            'batch_size': 8,
            'num_workers': 2
        },
        {
            'ds_path': 'data/puerto_rico.zarr',
            'ba_vars': ['vv', 'vh'],
            'aggregation': 'mean',
            'sat_orbit_state': 'dummy',
            'timestep_length': 1,
            'event_start_date': '20170920',
            'event_end_date': '20170921',
            'input_vars': ['vv_before', 'vv_after', 'vh_before', 'vh_after'],
            'target': 'landslides',
            'include_negatives': False,
            'split_fp': 'data/puerto_rico_70_20_10.yaml',
            'batch_size': 8,
            'num_workers': 2
        }
    ])

    encoder, models = instantiate_da_models(smp.UnetPlusPlus, 'resnet18', num_channels=4, classes=1)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    discriminator = MLPDiscriminator(512, [512], 3).to(device)
    models = [model.to(device) for model in models]

    da_trainer_module = plDATrainerModule(
        encoder,
        models,
        discriminator,
        model_losses=['ce', 'ce', 'ce'],
        lr=1e-3,
        device=device,
    )

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(da_trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')

