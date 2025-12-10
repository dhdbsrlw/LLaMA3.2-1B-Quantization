import os
import torch
import random
from torch.utils.data import Dataset

class TinyStoriesTrainDataset(Dataset):
    def __init__(self, config, tokenizer, prompter, split: str = "train", min_input_tokens: int = 20, min_output_tokens: int = 20):

        self.config = config
        self.tokenizer = tokenizer
        self.prompter = prompter

        self.split = split
        self.min_input_tokens = min_input_tokens
        self.min_output_tokens = min_output_tokens

        if split == "train":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
        elif split == "val":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
        else:
            raise ValueError(f"Unknown split (for training): {split}")

        stories = [
            story.strip()
            for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>")
            if story.strip()
        ]

        self.data = stories
        self.length = len(stories)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        story = self.data[idx]
        input_text, output_text = self._make_input_output(story)

        data_point = {
            "input": input_text,
            "output": output_text,
        }

        # 2) apply the same function you used with HF .map
        tokenized = self.generate_and_tokenize_prompt(data_point)
        return tokenized


    def _make_input_output(self, story: str):
        """Random token-level split into (input, output)."""
        tokens = self.tokenizer.encode(story, add_special_tokens=False)

        # If too short, just return whole story as input, empty output (edge case)
        if len(tokens) < self.min_input_tokens + self.min_output_tokens:
            given_tokens = tokens[:-self.min_output_tokens] or tokens
            generated_tokens = tokens[len(given_tokens):]
        else:
            split_tok = random.randint(
                self.min_input_tokens,
                len(tokens) - self.min_output_tokens,
            )
            given_tokens = tokens[:split_tok]
            generated_tokens = tokens[split_tok:]

        input_text = self.tokenizer.decode(given_tokens).strip()
        output_text = self.tokenizer.decode(generated_tokens).strip()
        return input_text, output_text


    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = result["input_ids"]

        # add eos if needed
        if (
            add_eos_token
            and len(input_ids) < self.config.max_seq_len
            and input_ids[-1] != self.tokenizer.eos_token_id
        ):
            input_ids.append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["input_ids"] = input_ids
        # labels initially = input_ids (will mask later if needed)
        result["labels"] = input_ids.copy()

        return result


    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["input"],
            data_point["output"],
        )

        tokenized_full_prompt = self.tokenize(
                                full_prompt, 
                                add_eos_token=self.config.add_eos_token)

        # mask inputs
        if self.config.mask_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["input"], None # data_point["output"]
            )
            tokenized_user_prompt = self.tokenize(
                user_prompt, add_eos_token=False 
                # self.config.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # mask input portion
            labels = tokenized_full_prompt["labels"]
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + labels[user_prompt_len:]
            )

        return tokenized_full_prompt





class TinyStoriesEvalDataset(Dataset):
    """Dataset class for TinyStories dataset."""

    def __init__(self, tokenizer, split: str = "train", seq_len: int = 128, nsamples: int = None):
        self.tokenizer = tokenizer
        self.split = split
        self.seq_len = seq_len

        if split == "train" or split == "calib":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
        elif split == "test":
            data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load raw stories
        stories = [
            story.strip()
            for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>")
            if story.strip()
        ]

        if nsamples is not None:
            stories = stories[:nsamples] 
            

        if split == "train":
            # keep per-story data for training
            self.data = [
                {"idx": idx, "text": text}
                for idx, text in enumerate(stories)
            ]
            self.length = len(self.data)

        else:  # split == "test" or split == "calib"
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
        else:  # test/calib split
            # Return a single [seq_len] tensor
            return self.sequences[idx]