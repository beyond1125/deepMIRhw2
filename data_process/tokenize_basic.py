"""Tokenize the beat-aligned POP909 training split for the Basic task (all tracks merged, REMI).

Input is the output of fix_timesig.py. Writes:
  <tokenizer_out>                 MidiTok tokenizer config (needed at inference time)
  <out_dir>/train.pkl, val.pkl    {'names': [...], 'ids': [np.int16 arrays]}
  <out_dir>/split.json            song ids in each split

Usage:
  python tokenize_basic.py --src ../../data/POP909_fixed/training \
      --out_dir ../../data/tokens_basic --tokenizer_out ../configs/tokenizer_remi.json
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from miditok import REMI, TokenizerConfig
from symusic import Score, Track


def build_tokenizer():
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        num_tempos=32,
        tempo_range=(50, 200),
        use_time_signatures=True,
        time_signature_range={4: [1, 2, 3, 4, 5, 6, 7, 8]},
        use_programs=False,
        special_tokens=['PAD', 'BOS', 'EOS'],
    )
    return REMI(config)


def merge_tracks(score):
    """Collapse all tracks into one piano track (Basic task merges MELODY, BRIDGE, PIANO)."""
    merged = Track(name='PIANO', program=0)
    for track in score.tracks:
        merged.notes.extend(track.notes)
    merged.notes.sort(key=lambda n: (n.time, n.pitch))
    score.tracks.clear()
    score.tracks.append(merged)
    return score


def encode(tokenizer, path):
    score = merge_tracks(Score(str(path)))
    seq = tokenizer.encode(score)
    seq = seq[0] if isinstance(seq, list) else seq
    return np.array([tokenizer['BOS_None']] + seq.ids + [tokenizer['EOS_None']], dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tokenizer_out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    tokenizer = build_tokenizer()
    Path(args.tokenizer_out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer_out)

    songs = sorted(p.name for p in Path(args.src).iterdir() if p.is_dir())
    rng = random.Random(args.seed)
    val = set(rng.sample(songs, round(len(songs) * args.val_ratio)))
    splits = {'train': [s for s in songs if s not in val], 'val': sorted(val)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bar_id = tokenizer['Bar_None']
    for split, names in splits.items():
        ids = [encode(tokenizer, Path(args.src) / s / f'{s}.mid') for s in names]
        with open(out_dir / f'{split}.pkl', 'wb') as f:
            pickle.dump({'names': names, 'ids': ids}, f)
        lens = np.array([len(x) for x in ids])
        tok_per_bar = np.concatenate([[len(x) / max(1, (x == bar_id).sum())] for x in ids])
        print(f'{split}: {len(ids)} songs, {lens.sum():,} tokens, len mean {lens.mean():.0f} '
              f'max {lens.max()}, tokens/bar mean {tok_per_bar.mean():.1f} p95 {np.percentile(tok_per_bar, 95):.1f}')
    with open(out_dir / 'split.json', 'w') as f:
        json.dump(splits, f, indent=1)
    print('vocab size:', len(tokenizer))


if __name__ == '__main__':
    main()
