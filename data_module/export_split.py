from typing import Optional, Tuple
from collections import defaultdict
import yaml
import argparse

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher

from data_module.base import before_after_ds, batching_dataset, BeforeAfterDatasetBatches
from utils import seed_everything

parser = argparse.ArgumentParser()

parser.add_argument("--ds_path", type=str, default='data/hokkaido_japan.zarr')
parser.add_argument("--ba_vars", type=str, nargs='*', default=['vv', 'vh'])
parser.add_argument("--timestep_length", type=int, default=4)
parser.add_argument("--event_start_date", type=str, default='20180905')
parser.add_argument("--event_end_date", type=str, default='20180907')
parser.add_argument("--input_vars", type=str, nargs='+', default=['vv_before', 'vv_after', 'vh_before', 'vh_after'])
parser.add_argument("--include_negatives", type=bool, default=False)
parser.add_argument("--target", type=str, default='landslides')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[0.7, 0.2, 0.1])
parser.add_argument("--out_file", type=str, default='data/splits.yaml')


args = parser.parse_args()

seed_everything(args.seed)

ba_ds = before_after_ds(
    ds_path=args.ds_path,
    ba_vars=args.ba_vars,
    aggregation='mean', # doesn't matter for this, dummy input
    timestep_length=args.timestep_length,
    event_start_date=args.event_start_date,
    event_end_date=args.event_end_date
    )

batches, mean_std_dict = batching_dataset(ba_ds, args.input_vars, args.target, args.include_negatives)

n = len(batches)
gen = torch.Generator().manual_seed(args.seed)
train_split, val_split, test_split = random_split(range(n), args.train_val_test_split, gen)
out_yaml = {
    'train': sorted(train_split.indices),
    'val': sorted(val_split.indices),
    'test': sorted(test_split.indices),
}
yaml.safe_dump(out_yaml, args.out_file)
