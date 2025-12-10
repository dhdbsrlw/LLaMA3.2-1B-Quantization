# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/quantize_gptq.py

# Legacy: 삭제예정 (TODO)

import os, json
import argparse
import logging
import random
import torch
import numpy as np
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

from src.utils import set_seed

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_calibration_dataset(seed, tokenizer, seq_len, nsamples=128, buffer_size=128): # buffer_size=5000):
    set_seed(seed)

    # load raw dataset 
    data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    raw_data = [
            story.strip()
            for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>")
            if story.strip()
        ]

    # parse N samples
    raw_data = random.sample(raw_data, k=nsamples)

    # concaenate long enough text
    all_text = "\n\n".join(
        raw_data[:buffer_size]
    )  # further reduce sample size (large enough to cover nsamples * seq_len tokens)

    encoding = tokenizer(all_text, return_tensors="pt")

    # gather dataset
    dataset = []
    for _ in range(nsamples):
        # pick a random starting index
        i = random.randint(0, encoding.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        input_ids = encoding.input_ids[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})

    return dataset


def quantize(base_model, tokenizer_name=None, quantized_model_dir=None):
    if tokenizer_name is None:
        tokenizer_name = base_model

    if quantized_model_dir is None:
        quantized_model_dir = f"{base_model.split('/')[-1]}-GPTQ"

    logging.info(f"base_model = {base_model}, tokenizer = {tokenizer_name}")

    # base quantization config example 
    quantize_config = BaseQuantizeConfig(
        bits=4,          # quantize model to 4-bit
        group_size=128,  # recommended to set the value to 128
        desc_act=True,   # set to False can significantly speed up inference, but the perplexity may slightly degrade
    )  

    # load un-quantized model, by default (The model will always be loaded into CPU memory.)
    model = AutoGPTQForCausalLM.from_pretrained(base_model, quantize_config)

    # get calibration data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    examples = get_calibration_dataset(
        nsamples=128,
        seed=0,
        seq_len=2048,
        tokenizer=tokenizer,
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    print(
        'quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"\n'
    )
    model.quantize(examples, use_triton=False)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    tokenizer.save_pretrained(quantized_model_dir)

    print("Quantization Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/nas2/checkpoints/Llama-3.2-1B", help="path to the base model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="if None, base model name is used")
    parser.add_argument(
        "--quantized_model_dir",
        type=str,
        default="/nas2/checkpoints/hf_cache_yj/Llama-3.2-1B-GPTQ-4bit",
        # default=None,
        help="if None, it is inferred from model_path (base_model).",
    )

    args = parser.parse_args()

    model = quantize(
        base_model=args.model_path,
        tokenizer_name=args.tokenizer_path,
        quantized_model_dir=args.quantized_model_dir,
    )