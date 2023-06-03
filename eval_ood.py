import torch
import utils
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from trainer_module.supervised_trainer import plSupervisedTrainerModule
from models.da_models import instantiate_da_models
from data_module.single_dm import SingleBeforeAfterCubeDataModule

if __name__ == '__main__':
    utils.seed_everything(0)
    
    state_dict_paths = [""]
    encoder, models = instantiate_da_models(smp.UnetPlusPlus, 'resnet18', num_heads=1, num_channels=4, classes=2)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = models[0].to(device)

    val_module = plSupervisedTrainerModule(
        model=model
    )

    dm = SingleBeforeAfterCubeDataModule(
        **{
            'ds_path': 'data/talakmau_indonesia.zarr',
            'ba_vars': ['vv', 'vh'],
            'aggregation': 'mean',
            'sat_orbit_state': 'd',
            'timestep_length': 1,
            'event_start_date': '20220225',
            'event_end_date': '20220226',
            'input_vars': ['vv_before', 'vv_after', 'vh_before', 'vh_after'],
            'target': 'landslides',
            'include_negatives': False,
            'split_fp': 'data/talakmau_70_20_10.yaml',
            'batch_size': 32,
            'num_workers': 4
        }
    )

    trainer = pl.Trainer()
    trainer.test(val_module, datamodule=dm)