from trainer_module.da_trainer import plDATrainerModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from models.da_models import instantiate_da_models
from models.discriminator import MLPDiscriminator
from data_module.multi_dm import MultiBeforeAfterCubeDataModule
from trainer_module.helpers import FocalTverskyLoss
import utils

from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dm_configs', nargs='+', type=str, default=['config/hokkaido.yaml', 'config/kaikoura.yaml', 'config/puerto_rico.yaml'])
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--model_lambdas', nargs='+', type=float, default=None)
    parser.add_argument('--disc_lambda', type=float, default=1)
    parser.add_argument('--da_lambda', type=float, default=1)
    args = parser.parse_args()

    utils.seed_everything(0)
    
    # Load config files
    configs = [utils.parse_yaml(config_path) for config_path in args.dm_configs]

    dm = MultiBeforeAfterCubeDataModule(configs)

    n_channels = len(configs[0]['input_vars'])
    encoder, models = instantiate_da_models(smp.UnetPlusPlus, args.encoder, num_channels=n_channels, classes=1)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    encoder_out_size = utils.get_encoder_output_channels(args.encoder)
    discriminator = MLPDiscriminator(encoder_out_size, [encoder_out_size], len(models)).to(device)
    models = [model.to(device) for model in models]

    if args.model_lambdas is None:
        args.model_lambdas = [1 for _ in range(len(models))] # more than needed, but keep for compatibility w/ list ordering
    
    da_trainer_module = plDATrainerModule(
        encoder,
        models,
        discriminator,
        model_losses={
            0: FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75),
        },
        lr=args.lr,
        device=device,
        model_lambdas=args.model_lambdas,
        disc_lambda=args.disc_lambda,
        da_lambda=args.da_lambda,
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val/0/jaccard', mode='max')

    trainer = pl.Trainer(max_epochs=args.n_epochs, val_check_interval=0.5)
    trainer.fit(da_trainer_module, datamodule=dm)
    trainer.validate(datamodule=dm, ckpt_path='best')
    trainer.test(datamodule=dm, ckpt_path='best')