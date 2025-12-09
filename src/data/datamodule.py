# TODO: 삭제 예정

"""LightningDataModule` for the TinyStories dataset."""

import glob
import os
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from data.dataclass import TinyStoriesDataset

class TinyStoriesDataModule(LightningDataModule):
    """`LightningDataModule` for the TinyStories dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download, split, transform and process the data.
    """

    def __init__(
        self,
        batch_size: int = 8,
        seq_len: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        # self.data_train = PretokDataset(split="train", max_seq_len=self.hparams.seq_len)
        # self.data_val = PretokDataset(split="val", max_seq_len=self.hparams.seq_len)
        # self.data_test = PretokDataset(split="test", max_seq_len=self.hparams.seq_len)

        self.data_train = TinyStoriesDataset(split="train")
        self.data_val = TinyStoriesDataset(split="val")
        # TODO: tokenize
        # TODO: max_seq_len


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
        )


    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the test dataloader.

    #     :return: The test dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #     )

    # def teardown(self, stage: Optional[str] = None) -> None:
    #     """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
    #     `trainer.test()`, and `trainer.predict()`.

    #     :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #         Defaults to ``None``.
    #     """
    #     pass

    # def state_dict(self) -> dict[Any, Any]:
    #     """Called when saving a checkpoint. Implement to generate and save the datamodule state.

    #     :return: A dictionary containing the datamodule state that you want to save.
    #     """
    #     return {}

    # def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.

    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """
    #     pass


if __name__ == "__main__":
    _ = TinyStoriesDataModule()