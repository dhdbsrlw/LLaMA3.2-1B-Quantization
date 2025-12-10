# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/quantize_gptq.py

# GPTQ
# conda activate edge

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
from gptqmodel import GPTQModel, QuantizeConfig

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
        quantized_model_dir = f"{base_model.split('/')[-1]}-gptqmodel-4bit"

    logging.info(f"base_model = {base_model}, tokenizer = {tokenizer_name}")

    # GPTQModel quantization config
    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        # e.g. gptaq=True if you want experimental GPTAQ
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # your calibration dataset (already good)
    examples = get_calibration_dataset(
        nsamples=128,
        seed=0,
        seq_len=2048,
        tokenizer=tokenizer,
    )

    # optional: squeeze batch dim so it's closer to their examples
    calib_data = [
        {k: v.squeeze(0) for k, v in ex.items()}
        for ex in examples
    ]

    # load FP model with quant config
    model = GPTQModel.load(base_model, quant_config, device="cuda")

    # quantize
    model.quantize(calib_data, batch_size=1)

    # save quantized model
    model.save(quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)

    print("Quantization Done.")
    return model


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