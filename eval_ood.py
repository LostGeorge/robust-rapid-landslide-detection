import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import argparse

from sar_unet_src.datamodule import BeforeAfterCubeDataModule
from sar_unet_src.lit_module import plUNET
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument("--ds_path", type=str, default='data/talakmau_indonesia.zarr')
    parser.add_argument("--ba_vars", type=str, default='vv,vh')
    parser.add_argument("--event_start_date", type=str, default='20220225')
    parser.add_argument("--event_end_date", type=str, default='20220226')
    parser.add_argument("--timestep_length", type=int, default=4)
    parser.add_argument("--input_vars", type=str, default='vv_before,vv_after,vh_before,vh_after')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    args.ba_vars = args.ba_vars.split(',')
    args.input_vars = args.input_vars.split(',')

    utils.seed_everything(0)
    
    model = plUNET(num_channels=len(args.input_vars))
    dm = BeforeAfterCubeDataModule(
        ds_path=args.ds_path,
        ba_vars=args.ba_vars,
        aggregation='mean',
        timestep_length=args.timestep_length,
        event_start_date=args.event_start_date,
        event_end_date=args.event_end_date,
        input_vars=args.input_vars,
        target='landslides',
        train_val_test_split=(0.0, 0.0, 1.0),
        batch_size=args.batch_size,
        include_negatives=False,
        num_workers=args.num_workers
    )
    trainer = pl.Trainer()
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()
