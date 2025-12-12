# Modified from https://github.com/Nota-NetsPresso/shortened-llm/blob/main/src/lora_retrain.py

import argparse
import os
import math
import torch
import wandb
import torch.nn.functional as F
import transformers
import numpy as np
import pyrootutils 
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from src.LLMPruner.utils.prompter import StoryPrompter # Prompter, ZeroPrompter
from src.utils import get_model, set_seed, build_config
from src.data.dataclass import TinyStoriesTrainDataset 


def prepare_data(config, tokenizer, prompter):
    train_dataset = TinyStoriesTrainDataset(
        config=config,
        tokenizer=tokenizer,
        prompter=prompter,
        split="train",
    )
    val_dataset = TinyStoriesTrainDataset(
        config=config,
        tokenizer=tokenizer,
        prompter=prompter,
        split="val",
    )
    return train_dataset, val_dataset


class PerplexityCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        if "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            print(f"\nPerplexity: {ppl:.4f}\n")
            metrics["eval_ppl"] = ppl # logged to W&B automatically

            # save to wandb explicitly
            if state.is_world_process_zero:
                wandb.log(
                    {
                        "eval_loss": eval_loss,
                        "eval_ppl": ppl,
                        "epoch": metrics.get("epoch", None),
                        "step": state.global_step,
                    },
                    step=state.global_step,
                )


def prepare_trainer(config, model, tokenizer, train_data, val_data):
        
    # (1) set dtype
    fp16_flag = True
    bf16_flag = False

    if config.use_bfloat:
        model = model.bfloat16()
        fp16_flag = False
        bf16_flag = True

    # (2) compute misc
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    
    # (3) set training args
    args = transformers.TrainingArguments(
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        fp16=fp16_flag,
        bf16=bf16_flag,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=None,
        group_by_length=config.group_by_length,
        report_to="wandb",
        run_name=config.output_dir.split("/")[-1],
        metric_for_best_model="eval_ppl",
        greater_is_better=False,
        max_steps=config.max_steps,        
        # epochs will be ignored anyway if max_steps > 0
    )

    # (4) set trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # compute_metrics=compute_metrics,
        callbacks=[PerplexityCallback()],
    )

    return trainer


def main(config):
    # os.environ["WANDB_PROJECT"] = args.wandb_project 
    set_seed(config.seed)

    model, tokenizer, description = get_model(
        base_model=config.model_path,
        ckpt=config.ckpt,
        lora_ckpt=None,
        tokenizer=config.tokenizer_path,
        model_type=config.model_type,
        device=config.device,
        use_bfloat=config.use_bfloat,
    )
    print("Loaded model.")

    prompter = StoryPrompter()

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:  
            # https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36#662315ec5d73c1b9f90482ea
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Loaded tokenizer.")


    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules.split(","),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # Load Train Dataset & Trainer
    train_data, val_data = prepare_data(config, tokenizer, prompter)
    print("Loaded dataset.")

    trainer = prepare_trainer(config, model, tokenizer, train_data, val_data)
    print("Prepared Trainer.")


    # Start Training
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    print("Training completed.")

    model.state_dict = old_state_dict
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Model saving completed.")


    if config.save_lora_merge:
        model = None

        set_seed(config.seed)
        model, tokenizer, description = get_model(
            base_model=config.model_path,
            ckpt=config.ckpt,
            lora_ckpt=None,
            tokenizer=config.tokenizer_path,
            model_type=config.model_type,
            device=config.device,
            use_bfloat=config.use_bfloat,
        )

        from LLMPruner.peft import PeftModel

        lora_model = PeftModel.from_pretrained(
            model, config.output_dir, torch_dtype=torch.float16
        )

        print("LoRA Merged.")
        lora_model = lora_model.merge_and_unload()

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }
        model.save_pretrained(
            os.path.join(config.output_dir + "_lora_merge_fp16"),
            state_dict=deloreanized_sd,
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(os.path.join(config.output_dir + "_lora_merge_fp16"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg/train.yaml")
    args = parser.parse_args()

    config = build_config(cfg_path=args.cfg_path)
    print("# Loaded config.")
    
    main(config)