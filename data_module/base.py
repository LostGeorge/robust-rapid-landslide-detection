from typing import Optional, Tuple
from collections import defaultdict

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher


def before_after_ds(
    ds_path,
    sat_orbit_state,
    ba_vars,
    aggregation,
    timestep_length,
    event_start_date,
    event_end_date
    ):
    ds = xr.open_zarr(ds_path)
    for var in ba_vars:
        ds[var] = np.log(ds[var])
    
    if 'hokkaido' in ds_path:
        orbit_state_bool_arr = xr.DataArray(
            data=['d', 'd', 'a', 'd', 'd', 'd', 'a', 'd', 'd', 'a', 'd', 'd', 'a', 'd', 'd', 'a', 'd', 'd', 'a', 'd', 'a', 'd', 'd', 'a', 'd', 'd'],
            dims=['timestep']
        )
    elif 'talakmau' in ds_path:
        orbit_state_bool_arr = xr.DataArray(
            data=['a', 'd', 'a', 'd', 'a', 'd', 'a', 'd', 'a', 'd', 'a', 'd', 'a', 'd', 'a', 'a', 'a', 'a', 'a', 'a'],
            dims=['timestep']
        )
    elif 'kaikoura' in ds_path:
        orbit_state_bool_arr = xr.DataArray(
            data=['ascending', 'ascending', 'ascending', 'ascending', 'ascending'],
            dims=['timestep']
        )
    if 'puerto_rico' not in ds_path:
        ds = ds.where(orbit_state_bool_arr == sat_orbit_state, drop=True)
    
    ds = ds.drop_dims('timepair')
    ds = ds[['vv', 'vh', 'landslides']]
    before_ds = ds.sel(timestep=slice(None, event_start_date))
    after_ds = ds.sel(timestep=slice(event_end_date, None))

    if timestep_length < len(before_ds['timestep']):
        before_ds = before_ds.isel(timestep=range(-1 - timestep_length, -1))

    if timestep_length < len(after_ds['timestep']):
        after_ds = after_ds.isel(timestep=range(timestep_length))

    if aggregation == 'mean':
        before_ds = before_ds.mean(dim=('timestep'))
        after_ds = after_ds.mean(dim=('timestep'))
    elif aggregation == 'median':
        before_ds = before_ds.median(dim=('timestep'))
        after_ds = after_ds.median(dim=('timestep'))

    before_after_vars = []
    for suffix in ['before', 'after']:
        for var in ba_vars:
            before_after_vars.append(f'{var}_{suffix}')
    the_ds = before_ds.rename_vars({var: f'{var}_before' for var in ba_vars})
    for var in ba_vars:
        the_ds[f'{var}_after'] = after_ds[var]
    for var in the_ds.data_vars:
        the_ds[f'{var}_mean'] = the_ds[var].mean()
        the_ds[f'{var}_std'] = the_ds[var].std()
    map_vars = ['dem', 'dem_aspect', 'dem_curvature', 'dem_slope_radians', 'dem_slope_riserun', 'extraction_mask',\
                'landslides', 'vv_before', 'vv_after', 'vh_before', 'vh_after', 'old_landslides', 'reactivated_landslides']
    for var in map_vars:
        if var in the_ds.data_vars:
            the_ds[var] = the_ds[var].astype(np.float16) # need to do this or else kaikoura is too large to fit in memory
    return the_ds.load()


def batching_dataset(ds, input_vars, target, include_negatives):
    mean_std_dict = {}
    for var in input_vars:
        if not mean_std_dict.get(var):
            mean_std_dict[var] = {}
        mean_std_dict[var]['mean'] = ds[f'{var}_mean'].values
        mean_std_dict[var]['std'] = ds[f'{var}_std'].values

    batches = []
    bgen = xbatcher.BatchGenerator(ds, {'x': 128, 'y': 128})
    positives = 0
    negatives = 0
    for batch in bgen:
        positives_tmp = batch[target].sum().item()
        if not include_negatives and positives_tmp > 0:
            positives = positives + positives_tmp
            negatives += batch[target].size
            batches.append(batch)
        elif include_negatives and (batch['dem'] <= 0).sum() == 0:
            positives = positives + positives_tmp
            negatives += batch[target].size
            batches.append(batch)
    print(f"P/N", positives / negatives)
    return batches, mean_std_dict


class BeforeAfterDatasetBatches(Dataset):
    def __init__(self, batches, input_vars, target, mean_std_dict):
        print("**************** INIT CALLED ******************")
        self.batches = batches
        self.target = target
        self.input_vars = input_vars
        self.mean = np.stack([mean_std_dict[var]['mean'] for var in input_vars]).reshape((-1, 1, 1))
        self.std = np.stack([mean_std_dict[var]['std'] for var in input_vars]).reshape((-1, 1, 1))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        inputs = np.stack([batch[var].values for var in self.input_vars]).astype(np.float32)
        inputs = (inputs - self.mean) / self.std

        target = batch[self.target].values
        inputs = np.nan_to_num(inputs, nan=0)
        target = np.nan_to_num(target, nan=0)
        target = (target > 0)
        return inputs, target


