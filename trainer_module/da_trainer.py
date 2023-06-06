import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy
from ot import sliced_wasserstein_distance

from typing import Any, List, Union, Callable, Optional, Dict, Tuple
import itertools

from trainer_module.helpers import BinarySegmentationMetricsWrapper, domain_confusion_loss

class plDATrainerModule(pl.LightningModule):
    def __init__(
            self,
            encoder: torch.nn.Module,
            models: List[torch.nn.Module],
            discriminator: torch.nn.Module,
            model_losses: Dict[int, Callable],
            model_evals: Dict[int, int],
            lr: float,
            model_lambdas: Dict[int, float] = {0: 1},
            disc_lambda: float = 1,
            da_lambda: float = 1,
            d_delay: int = 3,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False, ignore=['encoder', 'models', 'discriminator'])
        self.encoder = encoder
        self.models = models
        self.discriminator = discriminator

        self.model_losses = model_losses
        self.model_evals = model_evals
        self.disc_loss = torch.nn.CrossEntropyLoss()
        self.da_loss = domain_confusion_loss
        
        self.metrics = torch.nn.ModuleDict({str(i): BinarySegmentationMetricsWrapper(device) for i in model_evals.keys()})
        self.disc_acc = Accuracy(task='multiclass', num_classes=len(models), average='micro')
        
        self.train_seg_losses = [[] for _ in range(len(models))]
        self.train_disc_losses = []
        self.train_da_losses = []

        self.train_enc_outs = [None for _ in range(len(models))]
        self.train_epoch_ct = 0


    def forward(self, inputs: List[torch.Tensor]):
        enc_out_stages = [self.encoder(x) for x in inputs] # List[List[tensor]]
        enc_out_embs = [torch.cat([F.adaptive_avg_pool2d(stages[j], 1)[:, :, 0, 0] \
             for j in range(1, len(stages))], dim=1) for stages in enc_out_stages]
        disc_out = self.discriminator(torch.cat(enc_out_embs, dim=0))
        dec_out = [self.models[self.model_evals[i]].decoder(*stages) for i, stages in enumerate(enc_out_stages)]
        seg_out = [self.models[self.model_evals[i]].segmentation_head(out)[:, 0, ...] for i, out in enumerate(dec_out)]
        return seg_out, disc_out, enc_out_embs

    def training_step(self, batch_dict: Dict[int, Tuple], batch_idx: int):
        '''
        Note: We assume batch_dict is idx -> input, label in the input order
        '''
        seg_out, disc_out, enc_out_embs = self([batch_dict[i][0] for i in range(len(batch_dict))])
        seg_losses = [loss_fn(seg_out[i], batch_dict[i][1].float()) for i, loss_fn in self.model_losses.items()]
        seg_loss = torch.sum(torch.stack(
            [seg_losses[i] * self.hparams.model_lambdas[i] for i in range(len(seg_losses))]))
        disc_labels = torch.cat([torch.ones(len(batch_dict[i][0])) * i for i in range(len(batch_dict))]).to(self.hparams.device)
        disc_loss = self.disc_loss(disc_out, disc_labels.long()) * self.hparams.disc_lambda
        da_loss = self.da_loss(disc_out) * self.hparams.da_lambda
        
        for i, enc_out in enumerate(enc_out_embs):
            enc_out_numpy = enc_out.detach().cpu().numpy()
            if self.train_enc_outs[i] is None:
                self.train_enc_outs[i] = enc_out_numpy
            else:
                self.train_enc_outs[i] = np.vstack((self.train_enc_outs[i], enc_out_numpy))
                
        self.log("train/disc_loss", disc_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/da_loss", da_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self.train_disc_losses.append(disc_loss.item())
        self.train_da_losses.append(da_loss.item())
        self.disc_acc.update(disc_out.detach(), disc_labels.long())
        
        for i in self.model_losses.keys():
            self.log(f"train/{i}/seg_loss", seg_losses[i].item(), on_step=True, on_epoch=False, prog_bar=True)
            self.train_seg_losses[i].append(seg_losses[i].item())

        for i, metrics_obj in self.metrics.items():
            idx = int(i)
            for metric_value in metrics_obj.metrics_dict['train_'].values():
                metric_value.update(seg_out[idx].detach(), batch_dict[idx][1].detach())

        seg_optimizer, seg_reg_optimizer, disc_optimizer, da_optimizer = self.optimizers()
        if self.train_epoch_ct >= self.hparams.d_delay:
            seg_reg_optimizer.zero_grad()
            self.manual_backward(seg_loss, retain_graph=True)
            seg_reg_optimizer.step()
            disc_optimizer.zero_grad()
            self.manual_backward(disc_loss, retain_graph=True)
            disc_optimizer.step()
            da_optimizer.zero_grad()
            self.manual_backward(da_loss)
            da_optimizer.step()
        else:
            seg_optimizer.zero_grad()
            self.manual_backward(seg_loss)
            seg_optimizer.step()

        return None # since manual optimization

    def on_train_epoch_end(self):
        print("======== Training Metrics ========")
        print(f"train/disc_loss", np.mean(self.train_disc_losses))
        print(f"train/disc_acc", self.disc_acc.compute().item())
        print(f"train/da_loss", np.mean(self.train_da_losses))
        self.train_disc_losses = []
        self.disc_acc.reset()
        self.train_da_losses = []
        for i, metrics_obj in self.metrics.items():
            idx = int(i)
            if idx in self.model_losses.keys():
                print(f"train/{i}/seg_loss", np.mean(self.train_seg_losses[idx]))
                self.train_seg_losses[idx] = []
            for metric_key, metric_value in metrics_obj.metrics_dict['train_'].items():
                print(f"train/{i}/{metric_key}", metric_value.compute().item())
                metric_value.reset()
        # seg_scheduler, disc_scheduler, da_scheduler = self.lr_schedulers()
        # seg_scheduler.step()
        # disc_scheduler.step()
        # da_scheduler.step()
        seg_scheduler, seg_reg_scheduler = self.lr_schedulers()
        seg_scheduler.step()
        seg_reg_scheduler.step()

        for i, j in itertools.combinations(range(len(self.models)), 2):
            w_dist = sliced_wasserstein_distance(self.train_enc_outs[i], self.train_enc_outs[j])
            print(f"{i} {j} wasserstein distance: {w_dist}")
        self.train_enc_outs = [None for _ in range(len(self.models))]
        self.train_epoch_ct += 1
        

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        if dataloader_idx not in self.model_evals.keys():
            return
        x, targets = batch
        preds = self.models[self.model_evals[dataloader_idx]](x)[:, 0, ...]

        for metric_value in self.metrics[str(dataloader_idx)].metrics_dict['val_'].values():
            metric_value.update(preds, targets.long())

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx not in self.model_evals.keys():
            return
        x, targets = batch
        preds = self.models[self.model_evals[dataloader_idx]](x)[:, 0, ...]

        for metric_value in self.metrics[str(dataloader_idx)].metrics_dict['test_'].values():
            metric_value.update(preds, targets.long())

    def on_validation_epoch_end(self):
        print("======== Validation Metrics ========")
        for i, metrics_obj in self.metrics.items():
            for metric_key, metric_value in metrics_obj.metrics_dict['val_'].items():
                print(f"val/{i}/{metric_key}", metric_value.compute().item())
                self.log(f"val/{i}/{metric_key}", metric_value.compute().item())
                metric_value.reset()

    def on_test_epoch_end(self):
        print("======== Test Metrics ========")
        for i, metrics_obj in self.metrics.items():
            for metric_key, metric_value in metrics_obj.metrics_dict['test_'].items():
                print(f"test/{i}/{metric_key}", metric_value.compute().item())
                self.log(f"test/{i}/{metric_key}", metric_value.compute().item())
                metric_value.reset()

    def configure_optimizers(self):
        seg_params = list(self.encoder.parameters())
        for i in self.model_losses.keys():
            seg_params.extend(list(self.models[i].decoder.parameters()))
            seg_params.extend(list(self.models[i].segmentation_head.parameters()))
        
        seg_optimizer = torch.optim.Adam(params=seg_params, lr=self.hparams.lr)
        seg_reg_optimizer = torch.optim.Adam(params=self.encoder.parameters(), lr=self.hparams.lr)
        disc_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.hparams.lr)
        da_optimizer = torch.optim.Adam(params=self.encoder.parameters(), lr=self.hparams.lr)
        
        seg_scheduler = torch.optim.lr_scheduler.StepLR(seg_optimizer, step_size=1, gamma=0.9)
        seg_reg_scheduler = torch.optim.lr_scheduler.StepLR(seg_reg_optimizer, step_size=1, gamma=0.9)
        # disc_scheduler = torch.optim.lr_scheduler.ConstantLR(disc_optimizer, factor=0.00001, total_iters=self.hparams.d_delay)
        # da_scheduler = torch.optim.lr_scheduler.ConstantLR(da_optimizer, factor=0.00001, total_iters=self.hparams.d_delay)

        # return [seg_optimizer, disc_optimizer, da_optimizer], [seg_scheduler, disc_scheduler, da_scheduler]
        return [seg_optimizer, seg_reg_optimizer, disc_optimizer, da_optimizer], [seg_scheduler, seg_reg_scheduler]


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
        model_losses=['bce', 'bce', 'bce'], # wrong but too lazy to fix
        lr=1e-3,
        device=device,
    )

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(da_trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')

