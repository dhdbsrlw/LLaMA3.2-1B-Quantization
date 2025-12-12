# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/anal_block_sensitivity_ppl.py

import argparse
import csv
import os
import time

import torch
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import count_params, get_block_pruned_network, get_model, set_seed, build_config
from src.data.dataset import get_loader
from src.eval.eval_ppl import llama_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg/block_analyze.yaml")
    args = parser.parse_args()

    config = build_config(cfg_path=args.cfg_path)
    print("# Loaded config.")

    assert config.model_type in ["pretrain", "pruneLLM", "tune_pruneLLM"], f"Unknown model_type: {config.model_type}"

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    unsorted_output_path = os.path.join(config.output_dir, "all_ppl_unsorted.csv")
    sorted_output_path = os.path.join(config.output_dir, "all_ppl_sorted.csv")
    block_order_path = os.path.join(config.output_dir, "block_order.csv")

    if not os.path.exists(block_order_path):
        model_orig, tokenizer, description = get_model(
            base_model=config.model_path,
            ckpt=config.ckpt,
            lora_ckpt=config.lora_ckpt,
            tokenizer=config.tokenizer_path,
            model_type=config.model_type,
            device="cpu",
            use_bfloat=config.use_bfloat,
        )
        print("# Loaded (original) model and tokenizer.")

        # Evaluate the model with a single block removal
        _, _, calib_loader = get_loader(tokenizer, 
                                # num_workers=config.num_workers,
                                # batch_size=config.batch_size, 
                                max_seq_len=config.max_seq_len,
                                nsamples=config.num_calib_data)
        print("# Loaded calibration dataset.\n")

        for block_idx in range(model_orig.config.__getattribute__("num_hidden_layers")):
            csv_log_path = os.path.join(
                config.output_dir, f"ppl_block{block_idx}_removed.csv"
            )
            if os.path.exists(csv_log_path):
                print(f"Already computed: {csv_log_path}")
                continue

            model = get_block_pruned_network(
                model_orig,
                unimportance_order=[block_idx],
                num_pruned_blocks=1,
                device=config.device,
            )

            # Measure PPL
            t0 = time.perf_counter()
            with torch.no_grad():
                ppl = llama_eval(model, calib_loader, config.device)
    
            # Save
            with open(csv_log_path, "w") as logfile:
                logwriter = csv.writer(logfile, delimiter=",")
                logwriter.writerow(
                    ["removed_block", "ppl_tinystories", "num_calib_data", "params"]
                )
                logwriter.writerow(
                    [block_idx, ppl, config.num_calib_data, count_params(model)]
                )

            print(f"PPL over (Train dataset) {config.num_calib_data} samples: {ppl}")
            print(f"  * time in sec: {time.perf_counter()-t0}")
            
            del model
        print("# Loaded calibration dataset.")

        # Collec the PPL-based block sensitivity results
        unsorted_results = []
        with open(unsorted_output_path, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["removed_block", "ppl_tinystories", "num_calib_data", "params"]
            )
            for block_idx in range(
                model_orig.config.__getattribute__("num_hidden_layers")
            ):
                csv_log_path = os.path.join(
                    config.output_dir, f"ppl_block{block_idx}_removed.csv"
                )
                with open(csv_log_path, "r") as file:
                    next(file)  # pass the header line
                    data = [float(i) for i in str(next(file).strip()).split(",")]
                    logwriter.writerow(data)
                    unsorted_results.append(data)

        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=False)

        block_order = []
        with open(sorted_output_path, "w") as logfile, open(
            block_order_path, "w"
        ) as logfile_order:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["removed_block", "ppl_tinystories", "num_calib_data", "params"]
            )
            logwriter.writerows(sorted_results)
            for data in sorted_results:
                block_order.append(int(data[0]))
            logwriter_order = csv.writer(logfile_order, delimiter=",")
            logwriter_order.writerow(block_order)

        print(f"=== block order removed: {block_order_path}")
        print(block_order)
        print(f"len: {len(block_order)}")

    else:
        print(f"Already Exist: {block_order_path}")