# Compression Ratio

import os
import argparse

def get_dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total

if __name__ == "__main__":

    base_model_dir = "/nas2/checkpoints/Llama-3.2-1B"
    quant_model_dir = "./checkpoints/Llama-3.2-1B-Block-Pruning-n3"

    if not os.path.exists(base_model_dir) or not os.path.exists(quant_model_dir):
        raise FileNotFoundError("One of the specified model directories does not exist.")

    base_size = get_dir_size(base_model_dir)
    quant_size = get_dir_size(quant_model_dir)

    compression_ratio = base_size / quant_size

    print(f"Base size: {base_size / (1024**3):.2f} GB")
    print(f"Quant size: {quant_size / (1024**3):.2f} GB")
    print(f"Compression ratio: {compression_ratio:.2f}x")