"""Basic Task b: continue each 8-bar TA prompt to a 32-bar piece (8 + 24 bars).

--prompt_dir is the TA folder (e.g. POP909/prompts/basic-b) whose sub-folders hold <id>.mid and
beat_midi.txt. Prompts are re-aligned to the beat grid exactly like the training data.

Usage:
  python continue_prompt.py --ckpt checkpoints/basic --prompt_dir POP909/prompts/basic-b \
      --config A_t1.0_p0.95 --output_dir out/task_b
"""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from symusic import Score

from data_process.fix_timesig import fix_one, load_beats
from sampling import encode_score, generate_complete, load, pick_device, sampling_config, save_midi


def load_prompt(tokenizer, folder):
    mid = next(p for p in sorted(folder.glob('*.mid')))
    beat_times, downbeats = load_beats(folder / 'beat_midi.txt')
    with tempfile.TemporaryDirectory() as tmp:
        fixed = Path(tmp) / mid.name
        fix_one(mid, beat_times, downbeats, fixed)
        return encode_score(tokenizer, Score(str(fixed)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--prompt_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', nargs='+', default=None,
                    help='one or more preset names in configs/sampling.json (default: all presets)')
    ap.add_argument('--bars', type=int, default=32, help='total bars including the prompt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load(args.ckpt, device)
    presets = args.config or list(json.load(open(Path(__file__).parent / 'configs' / 'sampling.json')))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = tokenizer['Bar_None']

    folders = sorted(p for p in Path(args.prompt_dir).iterdir() if p.is_dir())
    prompts = {f.name: load_prompt(tokenizer, f) for f in folders}
    meta = []
    for name in presets:
        cfg = sampling_config(name)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        seqs, retries = generate_complete(model, tokenizer, list(prompts.values()), args.bars, cfg, generator=gen)
        for (pid, prefix), ids, rt in zip(prompts.items(), seqs, retries):
            fname = f'prompt{pid}_{name}.mid'
            save_midi(tokenizer, ids, out_dir / fname)
            meta.append({'file': fname, 'prompt': pid, 'config': name, 'sampling': cfg,
                         'prompt_bars': prefix.count(bar), 'total_bars': ids.count(bar),
                         'discarded_attempts': rt})
        print(f'{name}: {cfg} done', flush=True)
    json.dump({'seed': args.seed, 'pieces': meta}, open(out_dir / 'generation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
