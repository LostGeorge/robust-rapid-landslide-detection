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
    
    # Load config files
    config_paths = ['config/hokkaido.yaml', 'config/kaikoura.yaml', 'config/puerto_rico.yaml']
    configs = [utils.parse_yaml(config_path) for config_path in config_paths]

    dm = MultiBeforeAfterCubeDataModule(configs)

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
        device=device,
    )

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(da_trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')