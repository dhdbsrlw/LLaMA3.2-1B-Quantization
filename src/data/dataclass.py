import os
import torch
from torch.utils.data import Dataset

class TinyStoriesDataset(Dataset):
    """Dataset class for TinyStories dataset."""

    def __init__(self, tokenizer, split: str = "train", seq_len: int = 128):
        self.tokenizer = tokenizer
        self.split = split
        self.seq_len = seq_len

        if split == "train":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
        elif split == "val":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load raw stories
        stories = [
            story.strip()
            for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>")
            if story.strip()
        ]

        if split == "train":
            # keep per-story data for training
            self.data = [
                {"idx": idx, "text": text}
                for idx, text in enumerate(stories)
            ]
            self.length = len(self.data)

        else:  # split == "val"
            # 1) tokenize each story separately, then concatenate ids
            token_id_chunks = []
            for text in stories:
                ids = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]  # [Ti]
                token_id_chunks.append(ids)

            token_stream = torch.cat(token_id_chunks, dim=0)  # [T_total]

            # 2) prepend single BOS (LLaMA convention)
            bos_id = self.tokenizer.bos_token_id
            token_stream = torch.cat(
                (torch.tensor([bos_id], dtype=torch.long), token_stream),
                dim=0,
            )

            # 3) official PPL chunking
            num_tokens = token_stream.numel()
            num_sequences = num_tokens // self.seq_len

            trimmed = token_stream[: num_sequences * self.seq_len]
            sequences = trimmed.view(num_sequences, self.seq_len)  # [N, L]

            self.sequences = sequences
            self.length = num_sequences


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.split == "train":
            # you can later implement process_train_sample if needed
            return self.data[idx]
        else:  # val split
            # Return a single [seq_len] tensor
            return self.sequences[idx]