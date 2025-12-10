# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/eval_ppl.py

# conda activate edge
# python src/eval/eval_ppl.py 

import argparse
import csv
import os
import time
import yaml

import numpy as np
import torch
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tqdm import tqdm
from src.utils import count_params, get_model, set_seed, build_config
from src.data.dataset import get_loader

@torch.no_grad()
def llama_eval(model, test_loader, device):
    nlls = []
    for batch in tqdm(test_loader, desc="Evaluating PPL"):
        
        # print("batch shape:", batch.shape)  # batch shape: torch.Size([8, 128]) = (bsz, seq_len)
        # print("batch type:", batch.dtype) # batch type: torch.int64
        # print()

        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)

    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


# Evaluate only on TinyStories text
def eval_ppl(
    model,
    tokenizer,
    config,
):
    csv_header = []
    csv_value = []

    t0 = time.perf_counter()
    _, test_loader, _ = get_loader(tokenizer, 
                                num_workers=config.num_workers,
                                batch_size=config.batch_size, 
                                max_seq_len=config.max_seq_len)
    
    ppl = llama_eval(model, test_loader, config.device)
    
    elapsed = time.perf_counter() - t0
    print(f"PPL: {ppl:.4f} | time: {elapsed:.1f}s")

    # Logging
    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    nparams = count_params(model)
    print(f"Params: {nparams}")

    csv_header.append(f"PPL (Perplexity)")
    csv_value.append(ppl)

    if config.file_name is not None:
        csv_log_path = os.path.join(config.output_dir, f"{config.file_name}.csv")
    else:
        csv_log_path = os.path.join(config.output_dir, "ppl.csv") 

    with open(csv_log_path, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(csv_header + ["Params", "Memory", "Time(s)"])
        logwriter.writerow(csv_value + [nparams, mem, elapsed])


def generate_txt(
    model,
    tokenizer,
    config,
    # output_dir,
    # input_prompt,
    # generation_config,
    # num_output=5,
    # device="cuda",
):
    # Generate
    input_ids = tokenizer(config.input_prompt, return_tensors="pt")["input_ids"].to(config.device)
    input_len = input_ids[0].size(0)
    
    txt_path = os.path.join(config.output_dir, "gen_text", f"gen_text_{config.file_name}.txt") 
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("=== input ===\n")
        f.write(f"{config.input_prompt}\n")

    for i in range(config.num_output):
        with torch.no_grad():
            generation_output = model.generate(
                input_ids,
                **config.generation_config,
                # max_length=(input_len + max_seq_len),
                # min_length=(
                #     input_len + max_seq_len
                # ),  # forced output length (to avoid <EOS> sampling)
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output_len = len(s)
        output = tokenizer.decode(s)

        print(f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n")
        print(output)

        with open(txt_path, "a", encoding="utf8") as f:
            f.write(
                f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n"
            )
            f.write(f"{output}\n")


if __name__ == "__main__":

    # TODO: replace print with logging

    # (1) setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg/eval/ppl.yaml")
    args = parser.parse_args()

    config = build_config(cfg_path=args.cfg_path)
    print("# Loaded config.")

    assert config.model_type in ["pretrain", "pruneLLM", "tune_pruneLLM"]
    set_seed(config.seed)
    model, tokenizer, description = get_model(
        base_model=config.model_path,
        tokenizer=config.tokenizer_path,
        ckpt=config.ckpt,
        lora_ckpt=config.lora_ckpt,
        model_type=config.model_type,
        device=config.device,
        use_bfloat=config.use_bfloat,
    )
    print("# Loaded model and tokenizer.")

    # (2) main
    os.makedirs(config.output_dir, exist_ok=True)
    eval_ppl(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    print("# Evaluating PPL Done.")

    if config.input_prompt is not None:
        generate_txt(
            model=model,
            tokenizer=tokenizer,
            config=config,
            # output_dir=config.output_dir,
            # input_prompt=config.input_prompt,
            # generation_config=config.generation_config,
            # num_output=config.num_output,
            # device=config.device,
        )
        print("# Generating text Done.")
