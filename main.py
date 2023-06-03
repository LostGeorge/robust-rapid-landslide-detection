from trainer_module.da_trainer import plDATrainerModule
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from models.da_models import instantiate_da_models
from models.discriminator import MLPDiscriminator
from data_module.multi_dm import MultiBeforeAfterCubeDataModule
import utils


if __name__ == '__main__':
    utils.seed_everything(0)

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
            'batch_size': 32,
            'num_workers': 4
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
            'batch_size': 32,
            'num_workers': 4
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
            'batch_size': 32,
            'num_workers': 4
        }
    ])

    encoder, models = instantiate_da_models(smp.UnetPlusPlus, 'resnet18', num_channels=4, classes=2)
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
    )

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(da_trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')