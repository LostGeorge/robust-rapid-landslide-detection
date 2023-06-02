from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Union
from collections import defaultdict
from torch.utils.data.dataloader import _BaseDataLoaderIter, _collate_fn_t, _worker_init_fn_t
import yaml

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher

from data_module.base import before_after_ds, batching_dataset, BeforeAfterDatasetBatches
from data_module.single_dm import SingleBeforeAfterCubeDataModule


class MultiBeforeAfterCubeDataModule(LightningDataModule):
    """LightningDataModule.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            dm_args: List[Dict]
    ):
        super().__init__()
        self.dms: List[SingleBeforeAfterCubeDataModule] = []
        for dm_arg in dm_args:
            self.dms.append(SingleBeforeAfterCubeDataModule(**dm_arg))
        self.data_train = False
        self.data_val = False
        self.data_test = False

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            for dm in self.dms:
                dm.setup(stage=stage)
            self.data_train = True
            self.data_val = True
            self.data_test = True
                

    def train_dataloader(self) -> DataLoader:
        loader_dict = {i: DataLoader(
            dataset=dm.data_train,
            batch_size=dm.hparams.batch_size,
            num_workers=dm.hparams.num_workers,
            pin_memory=dm.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(dm.hparams.num_workers > 0)
        ) for i, dm in enumerate(self.dms)}
        # return InfiniteMultiDataLoader(loaders)
        combined_loader = CombinedLoader(loader_dict, mode='max_size_cycle')
        return combined_loader

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [DataLoader(
            dataset=dm.data_val,
            batch_size=dm.hparams.batch_size,
            num_workers=dm.hparams.num_workers,
            pin_memory=dm.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(dm.hparams.num_workers > 0)
        ) for dm in self.dms]
        return loaders

    def test_dataloader(self):
        loaders = [DataLoader(
            dataset=dm.data_test,
            batch_size=dm.hparams.batch_size,
            num_workers=dm.hparams.num_workers,
            pin_memory=dm.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(dm.hparams.num_workers > 0)
        ) for dm in self.dms]
        return loaders

class InfiniteMultiDataLoader(DataLoader):

    DUMMY_LEN=1e6
    
    def __init__(self, dataloaders: List[DataLoader], **kwargs):
        super().__init__(**kwargs)
        self.dataloaders = dataloaders
        self.generators = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        self.generators = [iter(dataloader) for dataloader in self.dataloaders]
        return self
    
    def __len__(self):
        return self.DUMMY_LEN

    def __next__(self):
        batch_contents = []
        for i in range(len(self.generators)):
            batch = next(self.generators[i], None)
            if batch is None:
                self.generators[i] = iter(self.dataloaders[i])
                batch = next(self.generators[i], None)
            batch_contents.append(batch)
        return batch_contents # [(x1, y1), (x2, y2), etc.]

if __name__ == '__main__':
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
            'batch_size': 64,
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
            'batch_size': 64,
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
            'batch_size': 64,
            'num_workers': 4
        }
    ])
    dm.prepare_data()
    dm.setup()

    train_dict = next(iter(dm.train_dataloader()))
    train_data_1, train_label_1 = train_dict[0]
    train_data_2, train_label_2 = train_dict[1]
    train_data_3, train_label_3 = train_dict[2]

    val_loaders = dm.val_dataloader()
    val_data_1, val_label_1 = next(iter(val_loaders[0]))
    val_data_2, val_label_2 = next(iter(val_loaders[1]))
    val_data_3, val_label_3 = next(iter(val_loaders[2]))

    test_loaders = dm.val_dataloader()
    test_data_1, test_label_1 = next(iter(test_loaders[0]))
    test_data_2, test_label_2 = next(iter(test_loaders[1]))
    test_data_3, test_label_3 = next(iter(test_loaders[2]))

    print(train_data_1.dtype, train_data_1.shape, train_label_1.dtype, train_label_1.shape)
    print(train_data_2.dtype, train_data_2.shape, train_label_2.dtype, train_label_2.shape)
    print(train_data_3.dtype, train_data_3.shape, train_label_3.dtype, train_label_3.shape)
    print(val_data_1.dtype, val_data_1.shape, val_label_1.dtype, val_label_1.shape)
    print(val_data_2.dtype, val_data_2.shape, val_label_2.dtype, val_label_2.shape)
    print(val_data_3.dtype, val_data_3.shape, val_label_3.dtype, val_label_3.shape)
    print(test_data_1.dtype, test_data_1.shape, test_label_1.dtype, test_label_1.shape)
    print(test_data_2.dtype, test_data_2.shape, test_label_2.dtype, test_label_2.shape)
    print(test_data_3.dtype, test_data_3.shape, test_label_3.dtype, test_label_3.shape)



