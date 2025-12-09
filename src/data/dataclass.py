import os
import json
import torch

from torch.utils.data import Dataset, DataLoader

class TinyStoriesDataset(Dataset):
    """Dataset class for TinyStories dataset."""

    def __init__(self, tokenizer, split: str="train", seq_len: int=2048):
    # def __init__(self, data_path: str):
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path : str
            Path to the JSON file containing the dataset.
        """
        self.tokenizer = tokenizer
        self.split = split


        if split == "train":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
        elif split == "val":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        data = sorted([story.strip() for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>") if story.strip()])
        

        if self.split == "train":
            self.data = []
            for idx, text in enumerate(data):
                self.data.append({
                    "idx": idx,
                    "text": text,
                })
            self.length = len(self.data)

        elif self.split == "val":
            # 1. Concatenate all text into a continuous corpus
            corpus_text = "\n\n".join(text for text in data)

            # 2. Tokenize into a flat token stream (no special tokens)
            token_stream = tokenizer(
                corpus_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]   # shape: [T]

            # 3. Prepend a single BOS token (LLaMA convention)
            bos_id = tokenizer.bos_token_id
            token_stream = torch.cat(
                (torch.tensor([bos_id], dtype=torch.long), token_stream),
                dim=0
            )

            # 4. Compute number of full-length evaluation sequences
            num_tokens = token_stream.numel()
            num_sequences = num_tokens // seq_len

            # 5. Discard incomplete tail and reshape into [num_sequences, seq_len]
            token_matrix = token_stream[: num_sequences * seq_len]
            # sequence_batch = token_matrix.view(num_sequences, seq_len) # type: torch.Tensor
            self.data = token_matrix.view(num_sequences, seq_len) # type: torch.Tensor
            self.length = len(self.data)

        else:
            raise ValueError(f"Unknown split: {split}")


    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

