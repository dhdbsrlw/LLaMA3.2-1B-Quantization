# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/block_prune.py

import argparse
import os

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.eval_ppl import eval_ppl, generate_txt
from src.utils import get_model, set_seed, get_block_pruned_network, build_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg/block_prune.yaml")
    args = parser.parse_args()

    config = build_config(cfg_path=args.cfg_path)
    print("# Loaded config.")

    assert config.model_type in ["pretrain", "pruneLLM", "tune_pruneLLM"], f"Unknown model_type: {config.model_type}"
    set_seed(config.seed)

    model_orig, tokenizer, description = get_model(
        base_model=config.model_path,
        ckpt=config.ckpt,
        lora_ckpt=config.lora_ckpt,
        tokenizer=config.tokenizer_path,
        model_type=config.model_type,
        device="cpu",
        use_bfloat=config.use_bfloat,
    )
    print("# Loaded model and tokenizer.\n")


    # Load the precomputed block unimportance order
    unimportance_order = []
    with open(config.block_order_csv, "r") as file:
        unimportance_order = [int(i) for i in str(next(file).strip()).split(",")]
    if not config.no_plus_heuristic:
        last_block_index = model_orig.config.num_hidden_layers - 1
        keep_block_info = [
            0,
            1,
            2,
            3,
            last_block_index - 1,
            last_block_index,
        ]  # to keep first and last few blocks unpruned
        unimportance_order = [
            idx for idx in unimportance_order if idx not in keep_block_info
        ]

    # Block-level pruning
    model = get_block_pruned_network(
        model_orig,
        unimportance_order=unimportance_order,
        num_pruned_blocks=config.num_pruned_blocks,
        device=config.device,
        use_bfloat=config.use_bfloat,
    )

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir, max_shard_size="10GB")
    tokenizer.save_pretrained(config.output_dir)

    # Measure PPL
    if not config.skip_validation:
        score_dir = os.path.join(config.output_dir + "_score")
        os.makedirs(score_dir, exist_ok=True)
        eval_ppl(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        config.output_dir = score_dir
        generate_txt(
            model=model, tokenizer=tokenizer, device=config.device, config=config, 
        )