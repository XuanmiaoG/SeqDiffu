"""
Train a diffusion model on images.
"""

import argparse
import json
import torch
import os
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # Update args dynamically
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()

    if args.num_scales <= 0:
        raise ValueError("num_scales must be greater than 0 for multi-scale diffusion.")

    logger.log("### Creating data loader...")
    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )
    next(data)

    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )

    logger.log("### Creating model...")
    model, _ = create_model_and_diffusion(**args_to_dict(args))
    model.to(dist_util.dev())

    logger.log("### Creating multi-scale diffusion...")
    multi_scale_diffusions = []
    for scale in range(args.num_scales):
        scale_args = {f"scale{scale}_{k}": v for k, v in args_to_dict(args).items()}
        logger.log(f"### Scale {scale} configuration: {json.dumps(scale_args, indent=2)}")
        _, diffusion = create_model_and_diffusion(**scale_args)
        multi_scale_diffusions.append(diffusion)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}")

    schedule_samplers = [
        create_named_schedule_sampler(args.schedule_sampler, diffusion)
        for diffusion in multi_scale_diffusions
    ]

    logger.log(f"### Saving the hyperparameters to {args.checkpoint_path}/training_args.json")
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")
    TrainLoop(
        model=model,
        multi_scale_diffusions=multi_scale_diffusions,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_samplers,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
