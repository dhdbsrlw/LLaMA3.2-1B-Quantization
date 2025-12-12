import torch
from torch.utils.data import Dataset, DataLoader
from src.data.dataclass import TinyStoriesEvalDataset, TinyStoriesTrainDataset


def get_loader(tokenizer, num_workers=2, batch_size=8, max_seq_len=128, nsamples=None):
    
    train_dataset = TinyStoriesEvalDataset(tokenizer, split="train", seq_len=max_seq_len, nsamples=nsamples)
    test_dataset = TinyStoriesEvalDataset(tokenizer, split="test", seq_len=max_seq_len, nsamples=nsamples)
    # calib_dataset = TinyStoriesEvalDataset(tokenizer, split="calib", seq_len=max_seq_len, nsamples=nsamples)

    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
    print("train_dataloader Done.")

    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
    print("test_dataloader Done.")

    # TODO: sanity check
    # calib_dataloader = DataLoader(
    #         dataset=calib_dataset,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         shuffle=True,
    #         pin_memory=True,
    #     )
    # print("calib_dataloader Done.")

    # return train_dataloader, test_dataloader, calib_dataloader
    return train_dataloader, test_dataloader

