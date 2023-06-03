import torch
import utils
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from trainer_module.supervised_trainer import plSupervisedTrainerModule
from models.da_models import instantiate_da_models
from data_module.single_dm import SingleBeforeAfterCubeDataModule

if __name__ == '__main__':
    utils.seed_everything(0)
    config = utils.parse_yaml('config/talakmau.yaml')
    
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

    dm = SingleBeforeAfterCubeDataModule(**config)

    trainer = pl.Trainer()
    trainer.test(val_module, datamodule=dm)