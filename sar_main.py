'''
Code from Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes
by Boehm et. al 2022, NIPS workshop on Tackling Climate Change with Machine Learning.
https://github.com/iprapas/landslide-sar-unet
'''

from src.lit_module import plUNET, plMLP
from src.base_data_module import BeforeAfterCubeDataModule
import pytorch_lightning as pl
import torch
import argparse
import sys

import utils


# a function to read plUNET parameters as program arguments
def add_plUNET_args(parent_parser):
    parser = parent_parser.add_argument_group("plUNET")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default='ce')
    return parent_parser

# a function to read plMLP parameters as program arguments
def add_plMLP_args(parent_parser):
    parser = parent_parser.add_argument_group("plMLP")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default='mse')
    return parent_parser


def add_BeforeAfterCubeDataModule_args(parent_parser):
    parser = parent_parser.add_argument_group("BeforeAfterCubeDataModule")
    parser.add_argument("--ds_path", type=str, default='data/hokkaido_japan.zarr')
    parser.add_argument("--ba_vars", type=str, default='vv,vh')
    parser.add_argument("--timestep_length", type=int, default=4)
    parser.add_argument("--event_start_date", type=str, default='20180905')
    parser.add_argument("--event_end_date", type=str, default='20180907')
    parser.add_argument("--input_vars", type=str, default='vv_before,vv_after,vh_before,vh_after')
    parser.add_argument("--target", type=str, default='landslides')
    # parser.add_argument("--train_val_test_split", type=str, default='0.7_0.2_0.1')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default=None)
    # parser.add_argument("--include_negatives", type=bool, default=False)
    return parent_parser


if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser = add_plUNET_args(parser)
    parser = add_BeforeAfterCubeDataModule_args(parser)
    # add argument for max_epochs
    parser.add_argument("--max_epochs", type=int, default=10)
    # parse the arguments
    unet_hparams = parser.parse_args()

     # create the top-level parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser = add_plMLP_args(parser)
    parser = add_BeforeAfterCubeDataModule_args(parser)
    # add argument for max_epochs
    parser.add_argument("--max_epochs", type=int, default=10)
    mlp_hparams = parser.parse_args()

    # transform ba_vars to list
    unet_hparams.ba_vars = unet_hparams.ba_vars.split(',')
    # transform input_vars to list
    unet_hparams.input_vars = unet_hparams.input_vars.split(',')
    # transform train_val_test_split to tuple
    # hparams.train_val_test_split = tuple([float(x) for x in hparams.train_val_test_split.split('_')])


    # run the main function
    print(unet_hparams)
    print(mlp_hparams)
    utils.seed_everything(0)
    # create the datamodule from the appropriate parsed arguments
    dm = BeforeAfterCubeDataModule(
        ds_path=unet_hparams.ds_path,
        ba_vars=unet_hparams.ba_vars,
        aggregation='mean',
        timestep_length=unet_hparams.timestep_length,
        event_start_date=unet_hparams.event_start_date,
        event_end_date=unet_hparams.event_end_date,
        input_vars=unet_hparams.input_vars,
        target='landslides',
        train_val_test_split=(0.7, 0.2, 0.1),
        batch_size=unet_hparams.batch_size,
        include_negatives=False,
        num_workers=unet_hparams.num_workers
    )

    # create the plUNET from the appropriate parsed arguments
    model = plUNET(
        lr=unet_hparams.lr,
        weight_decay=unet_hparams.weight_decay,
        num_channels=len(unet_hparams.input_vars),
        loss=unet_hparams.loss
    )
    # model2 = plMLP(
    #     encoder=model.encoder,
    #     lr=mlp_hparams.lr,
    #     weight_decay=mlp_hparams.weight_decay,
    #     num_channels=len(mlp_hparams.input_vars),
    #     loss=mlp_hparams.loss
    # )

    # create the trainer with the appropriate parsed arguments
    trainer = pl.Trainer(max_epochs=unet_hparams.max_epochs)

    # train the model
    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)
    if unet_hparams.save_path is not None:
        torch.save(model.state_dict(), unet_hparams.save_path)

    # train the model
    # trainer.fit(model2, datamodule=dm)
    # trainer.validate(datamodule=dm)
    # trainer.test(datamodule=dm)
    # if mlp_hparams.save_path is not None:
    #     torch.save(model2.state_dict(), mlp_hparams.save_path)
