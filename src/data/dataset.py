import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataclass import TinyStoriesDataset

def get_loader(tokenizer, num_workers=2, batch_size=8, max_seq_len=128):
    train_dataset = TinyStoriesDataset(tokenizer, split="train", seq_len=max_seq_len)
    eval_dataset = TinyStoriesDataset(tokenizer, split="val", seq_len=max_seq_len)
    
    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

    eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    return train_dataloader, eval_dataloader

