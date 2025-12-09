import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataclass import TinyStoriesDataset

def get_loader(tokenizer, num_workers=2, batch_size=8, max_seq_len=2048):
    train_data = TinyStoriesDataset(split="train")
    val_data = TinyStoriesDataset(split="val")

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


# TODO: 삭제 예정


# class IndexDataset(Dataset):
#     def __init__(self, tensors):
#         self.tensors = tensors

#     def __getitem__(self, index):
#         return self.tensors[index]

#     def __len__(self):
#         return len(self.tensors)


# def process_data(dataset, tokenizer, seq_len):
#     """
#     Prepares a dataset for perplexity evaluation for LLaMA-style models.

#     Args:
#         dataset (list[dict]): Each dict must contain a "text" field.
#         tokenizer: HuggingFace tokenizer for LLaMA.
#         seq_len (int): Number of tokens per evaluation segment.

#     Returns:
#         IndexDataset: A dataset where each item is a fixed-length token sequence.
#     """

#     # 1. Concatenate all text into a continuous corpus
#     corpus_text = "\n\n".join(sample["text"] for sample in dataset)

#     # 2. Tokenize into a flat token stream (no special tokens)
#     token_stream = tokenizer(
#         corpus_text,
#         return_tensors="pt",
#         add_special_tokens=False,
#     ).input_ids[0]   # shape: [T]

#     # 3. Prepend a single BOS token (LLaMA convention)
#     bos_id = tokenizer.bos_token_id
#     token_stream = torch.cat(
#         (torch.tensor([bos_id], dtype=torch.long), token_stream),
#         dim=0
#     )

#     # 4. Compute number of full-length evaluation sequences
#     num_tokens = token_stream.numel()
#     num_sequences = num_tokens // seq_len
#     assert num_sequences == len(dataset), "Number of sequences does not match dataset size."

#     # 5. Discard incomplete tail and reshape into [num_sequences, seq_len]
#     token_matrix = token_stream[: num_sequences * seq_len]
#     sequence_batch = token_matrix.view(num_sequences, seq_len)

#     # 6. Wrap into dataset class
#     return IndexDataset(tensors=sequence_batch)

