"""Basic Task a: unconditional generation of 32-bar pieces.

Usage:
  python generate_basic.py --ckpt checkpoints/basic --config A_t1.0_p0.95 --n 20 --output_dir out/task_a
  python generate_basic.py --ckpt checkpoints/basic --temperature 1.1 --top_p 0.9 --output_dir out/custom
"""
import argparse
import json
from pathlib import Path

import torch

from sampling import generate_complete, load, pick_device, sampling_config, save_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint folder (contains tokenizer_remi.json)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default=None, help='preset name in configs/sampling.json')
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--top_k', type=int)
    ap.add_argument('--top_p', type=float)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--bars', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load(args.ckpt, device)
    cfg = sampling_config(args.config, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device {device}, config {args.config or "custom"}: {cfg}', flush=True)

    meta, k = [], 0
    while k < args.n:
        bs = min(args.batch_size, args.n - k)
        seqs, retries = generate_complete(model, tokenizer, [[tokenizer['BOS_None']]] * bs, args.bars, cfg, generator=gen)
        for ids, rt in zip(seqs, retries):
            name = f'{k:02d}.mid'
            save_midi(tokenizer, ids, out_dir / name)
            meta.append({'file': name, 'tokens': len(ids), 'bars': ids.count(tokenizer['Bar_None']),
                         'discarded_attempts': rt})
            k += 1
        print(f'{k}/{args.n} done', flush=True)
    json.dump({'config': args.config, 'sampling': cfg, 'seed': args.seed, 'pieces': meta},
              open(out_dir / 'generation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
