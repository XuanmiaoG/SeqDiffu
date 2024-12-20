"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import json
import time
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, num_scales=1)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # Load training configuration
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'r') as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating models and multi-scale diffusions...")
    model, diffusions = create_multi_scale_models_and_diffusions(
        num_scales=args.num_scales,
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=args.hidden_dim,
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    set_seed(args.seed2)

    logger.log("### Sampling...on", args.split)

    # Load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),
        loop=False
    )

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")

    all_test_data = []
    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            if idx % world_size == rank:  # Split data across nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder: dist.barrier()

    iterator = iter(all_test_data)
    for cond in iterator:
        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        # Multi-scale sampling
        samples = []
        for scale_idx, diffusion in enumerate(diffusions):
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            scale_samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap
            )
            samples.append(scale_samples[-1])

        # Combine results from multi-scale
        combined_sample = sum(samples) / len(samples)

        logits = model.get_logits(combined_sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover, word_lst_ref, word_lst_source = [], [], []
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                with open(out_path, 'a') as fout:
                    for recov, ref, src in zip(word_lst_recover, word_lst_ref, word_lst_source):
                        print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
            dist.barrier()

    print(f'### Total time: {time.time() - start_t:.2f}s')
    print(f'### Output written to {out_path}')

def create_multi_scale_models_and_diffusions(num_scales, **kwargs):
    """
    Create multiple models and diffusions for multi-scale processing.
    """
    models = []
    diffusions = []
    for scale_idx in range(num_scales):
        model, diffusion = create_model_and_diffusion(**kwargs)
        models.append(model)
        diffusions.append(diffusion)
    return models[0], diffusions

if __name__ == "__main__":
    main()
