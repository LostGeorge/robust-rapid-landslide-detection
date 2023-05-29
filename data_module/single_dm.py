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

from data_module.base import before_after_ds, batching_dataset, BeforeAfterDatasetBatches


class SingleBeforeAfterCubeDataModule(LightningDataModule):
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
            ds_path: str,
            ba_vars,
            aggregation,
            timestep_length,
            event_start_date,
            event_end_date,
            input_vars,
            target,
            include_negatives=False,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.ds = None
        self.batches = None
        self.mean_std_dict = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.ds = before_after_ds(self.hparams.ds_path, self.hparams.ba_vars, self.hparams.aggregation,
                                      self.hparams.timestep_length, self.hparams.event_start_date,
                                      self.hparams.event_end_date)
            self.batches, self.mean_std_dict = batching_dataset(self.ds, self.hparams.input_vars, self.hparams.target,
                                                                self.hparams.include_negatives)

            dataset = BeforeAfterDatasetBatches(self.batches, self.hparams.input_vars, self.hparams.target,
                                                        mean_std_dict=self.mean_std_dict)

            train_val_test_split = [int(len(dataset) * x) for x in self.hparams.train_val_test_split]
            train_val_test_split[2] = len(dataset) - train_val_test_split[1] - train_val_test_split[0]
            train_val_test_split = tuple(train_val_test_split)
            print("*" * 20)
            print("Train - Val - Test SPLIT", train_val_test_split)
            print("*" * 20)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=train_val_test_split,
            )

            print("*" * 20)
            print("Train - Val - Test LENGTHS", len(self.data_train), len(self.data_val), len(self.data_test))
            print("*" * 20)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(self.hparams.num_workers > 0)
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0)
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0)
        )