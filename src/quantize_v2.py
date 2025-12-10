# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/quantize_gptq.py

# AWQ
# conda activate edge_2 (transformers==4.47.1)

import os, json
import argparse
import logging
import random
import torch
import numpy as np
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from datasets import load_dataset
from transformers import AutoTokenizer
# from gptqmodel import GPTQModel, QuantizeConfig
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from awq import AutoAWQForCausalLM # autoawq

from src.utils import set_seed

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# AWQ expects text, not tensor.

def get_calibration_dataset(seed, nsamples=128, buffer_size=128): # buffer_size=5000):
    set_seed(seed)

    # load raw dataset 
    data_path = "/nas2/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    raw_data = [
            story.strip()
            for story in open(data_path, "r", encoding="utf-8").read().split("<|endoftext|>")
            if story.strip()
        ]

    # parse N samples
    nsamples = min(nsamples, len(raw_data))
    raw_data = random.sample(raw_data, k=nsamples)

    return raw_data

    # # concaenate long enough text
    # all_text = "\n\n".join(
    #     raw_data[:buffer_size]
    # )  # further reduce sample size (large enough to cover nsamples * seq_len tokens)

    # encoding = tokenizer(all_text, return_tensors="pt")

    # # gather dataset
    # dataset = []
    # for _ in range(nsamples):
    #     # pick a random starting index
    #     i = random.randint(0, encoding.input_ids.shape[1] - seq_len - 1)
    #     j = i + seq_len
    #     input_ids = encoding.input_ids[:, i:j]
    #     attention_mask = torch.ones_like(input_ids)
    #     dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})

    # return dataset


def quantize(base_model, tokenizer_name=None, quantized_model_dir=None):
    """
    AWQ quantization using AutoAWQForCausalLM.
    """
    if tokenizer_name is None:
        tokenizer_name = base_model

    if quantized_model_dir is None:
        quantized_model_dir = f"{base_model.split('/')[-1]}-AWQ-4bit"

    logging.info(f"[AWQ] base_model = {base_model}, tokenizer = {tokenizer_name}")

    # AWQ 4-bit config (W4, group size 128, GEMM backend)
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    # quant_config = QuantizeConfig(
    #     bits=4,
    #     group_size=128,
    #     # e.g. gptaq=True if you want experimental GPTAQ
    # )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        use_cache=False,
        device_map="auto",          # spread across GPUs if multiple
        trust_remote_code=True,
    )


    # your calibration dataset (already good)
    calib_data = get_calibration_dataset(
        seed=0,
        nsamples=128,
    )

    # quantize
    # model.quantize(calib_data, batch_size=1)
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,     # our TinyStories texts
        # max_calib_seq_len=2048,  # optional, default=512 in AWQ
    )

    # save quantized model
    # model.save(quantized_model_dir, safetensors=True)
    model.save_quantized(quantized_model_dir, safetensors=True)
    tokenizer.save_pretrained(quantized_model_dir)

    logging.info("Quantization Done.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/nas2/checkpoints/Llama-3.2-1B", help="path to the base model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="if None, base model name is used")
    parser.add_argument(
        "--quantized_model_dir",
        type=str,
        default="/nas2/checkpoints/hf_cache_yj/Llama-3.2-1B-AWQ-4bit",
        # default=None,
        help="if None, it is inferred from model_path (base_model).",
    )

    args = parser.parse_args()

    model = quantize(
        base_model=args.model_path,
        tokenizer_name=args.tokenizer_path,
        quantized_model_dir=args.quantized_model_dir,
    )