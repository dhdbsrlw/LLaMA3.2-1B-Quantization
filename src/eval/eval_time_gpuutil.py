# nvidia_smi_gpuutil.csv
# latency_throughput_gpuutil.csv
# gen_output_gpuutil.txt

import argparse
import csv
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime

import torch
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import count_params, build_config, get_model, set_seed


MONITORING_ACTIVE = True   # will be set in main
INTERVAL = 0.5             # seconds between nvidia-smi samples


def get_gpu_metrics(gpu_index: int):
    """
    Get metrics for a specific GPU:
    - temperature.gpu
    - utilization.gpu
    - utilization.memory
    - memory.used
    """
    command = (
        f"nvidia-smi -i {gpu_index} "
        "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used "
        "--format=csv,noheader,nounits"
    )
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    metrics = result.stdout.decode("utf-8").strip().split(", ")

    # temperature.gpu, utilization.gpu, utilization.memory, memory.used
    metrics = (
        [int(metrics[0])]
        + [int(metric.strip("%")) for metric in metrics[1:3]]
        + [int(metrics[3].strip(" MiB"))]
    )
    return metrics


def monitor_gpu(gpu_util_csv: str, gpu_index: int = 0):
    """
    Periodically query GPU metrics and write them to a CSV file.

    Writes columns:
    - timestamp
    - temperature.gpu
    - utilization.gpu [%]
    - utilization.memory [%]
    - memory.used [MiB]
    """
    global MONITORING_ACTIVE, INTERVAL

    with open(gpu_util_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "timestamp",
                "temperature.gpu",
                "utilization.gpu [%]",
                "utilization.memory [%]",
                "memory.used [MiB]",
            ]
        )

        while MONITORING_ACTIVE:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            gpu_metrics = get_gpu_metrics(gpu_index)
            csv_writer.writerow([timestamp] + gpu_metrics)
            csvfile.flush()
            time.sleep(INTERVAL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg/eval/time_gpuutil.yaml")
    args = parser.parse_args()

    config = build_config(cfg_path=args.cfg_path)
    print("# Loaded config.")

    # (1) Setup & model preparation
    os.makedirs(config.output_dir, exist_ok=True)

    set_seed(config.seed)

    model, tokenizer, description = get_model(
        base_model=config.model_path,
        ckpt=config.ckpt,
        lora_ckpt=config.lora_ckpt,
        tokenizer=config.tokenizer_path,
        model_type=config.model_type,
        device=config.device,
        use_bfloat=config.use_bfloat,
    )

    # (2) Prepare input for batched generation
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [config.input_prompt] * config.batch_size,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(config.device)
    input_len = inputs[0].size(0)

    # Paths
    gpu_util_csv = os.path.join(config.output_dir, f"{config.file_name}_nvidia_smi_gpuutil.csv")
    time_stat_csv = os.path.join(config.output_dir, f"{config.file_name}_latency_throughput_gpuutil.csv")
    gen_log_path = os.path.join(config.output_dir, f"{config.file_name}_gen_output_gpuutil.txt")
    open(gen_log_path, "w", encoding="utf8").close()  # clear previous gen log

    # (3) Start GPU utilization monitor
    MONITORING_ACTIVE = True
    monitoring_thread = threading.Thread(
        target=monitor_gpu,
        args=(gpu_util_csv, config.gpu_index),
    )
    monitoring_thread.start()

    time_list = []        # latency per batch (sec)
    throughput_list = []  # throughput per run (tokens/s)

    try:
        for i in range(config.num_all_runs):
            if "cuda" in config.device:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=inputs,
                        **config.generation_config,
                        return_dict_in_generate=True,
                    )
                ender.record()
                torch.cuda.synchronize()

                batch_time = starter.elapsed_time(ender) / 1000.0  # ms -> sec
            else:
                raise NotImplementedError("Only CUDA device is supported for timing.")

            # Skip warmup runs
            if i < config.num_warmup_runs:
                continue

            # Decode and optionally log generated text
            output = tokenizer.batch_decode(generation_output.sequences)
            for b_idx, s in enumerate(output):
                tmp_size = len(generation_output.sequences[b_idx]) - input_len
                with open(gen_log_path, "a", encoding="utf8") as f:
                    f.write(
                        f"=== output {i} b{b_idx} | len gen {tmp_size} + input {input_len}\n"
                    )
                    f.write(f"{s}\n")
                print(
                    f"=== output {i} b{b_idx} | len gen {tmp_size} + input {input_len}\n"
                )
                print(f"{s}\n")

            # 1) LATENCY (sec per batch)
            time_list.append(batch_time)

            # 2) THROUGHPUT (tokens/sec)
            # Here we assume fixed generated length = config.max_seq_len per sample
            generated_tokens_per_sample = config.max_seq_len
            tokens_generated = config.batch_size * generated_tokens_per_sample
            throughput = tokens_generated / batch_time
            throughput_list.append(throughput)

    finally:
        # Stop GPU monitor
        MONITORING_ACTIVE = False
        monitoring_thread.join()


    # (4) Summarize measurements
    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem:.2f} MiB")

    nparams = count_params(model)
    print(f"Params: {nparams}")

    nbatches = len(throughput_list)

    if nbatches > 0:
        time_mean = statistics.mean(time_list)
        time_std = statistics.pstdev(time_list)

        throughput_mean = statistics.mean(throughput_list)
        throughput_std = statistics.pstdev(throughput_list)
    else:
        time_mean = time_std = throughput_mean = throughput_std = 0.0

    print(f"[Latency]   mean={time_mean:.4f} sec/batch, std={time_std:.4f}")
    print(f"[Throughput] mean={throughput_mean:.2f} tokens/s, std={throughput_std:.2f}")
    print(f"[GPUUTIL]   logged to: {gpu_util_csv}")

    # Save stats to CSV
    with open(time_stat_csv, "w", newline="") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [
                "time_mean(sec)",
                "time_std",
                "th_mean(tokens/s)",
                "th_std",
                "mem(MiB)",
                "out_len",
                "in_len",
                "nbatches",
                "batchsz",
                "nparam",
            ]
        )
        logwriter.writerow(
            [
                time_mean,
                time_std,
                throughput_mean,
                throughput_std,
                mem,
                config.max_seq_len,
                input_len,
                nbatches,
                config.batch_size,
                nparams,
            ]
        )
        # Optional: dump per-run latencies on the third row
        logwriter.writerow(time_list)