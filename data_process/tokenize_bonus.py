"""Tokenize MELODY + PIANO of the beat-aligned training split into interleaved bonus-task windows.

Each song is cut into 16-bar windows (stride 4 bars); every window becomes one sequence laid out
with melody lookahead k (see bonus_utils.py). Uses the same train/val song split as the Basic task.

Usage:
  python data_process/tokenize_bonus.py --src ../data/POP909_fixed/training \
      --split ../data/tokens_basic/split.json --lookahead 1 \
      --out_dir ../data/tokens_bonus_k1 --tokenizer_out configs/tokenizer_remi_bonus.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from symusic import Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bonus_utils import add_chords, bar_chords, bar_lengths_of, build_tokenizer, interleave, track_bars  # noqa: E402
from data_process.fix_timesig import load_beats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--split', required=True, help='split.json from tokenize_basic.py')
    ap.add_argument('--lookahead', type=int, default=1)
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--chords', action='store_true', help='add a chord-planning token to every accompaniment bar')
    ap.add_argument('--raw_src', help='TA training folder with chord_midi.txt / beat_midi.txt (needed with --chords)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tokenizer_out', required=True)
    args = ap.parse_args()

    tokenizer = build_tokenizer(chords=args.chords)
    tokenizer.save(args.tokenizer_out)
    splits = json.load(open(args.split))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, names in splits.items():
        ids, win_names = [], []
        for s in names:
            score = Score(str(Path(args.src) / s / f'{s}.mid'))
            lengths = bar_lengths_of(score)
            mel = track_bars(tokenizer, score, 'MELODY', lengths)
            acc = track_bars(tokenizer, score, 'PIANO', lengths)
            if args.chords:
                raw = Path(args.raw_src) / s
                beat_times, downbeats = load_beats(raw / 'beat_midi.txt')
                acc = add_chords(tokenizer, acc, bar_chords(raw / 'chord_midi.txt', beat_times, downbeats, lengths))
            n = len(lengths)
            for st in range(0, max(1, n - args.window + 1), args.stride):
                ids.append(interleave(tokenizer, mel[st:st + args.window], acc[st:st + args.window],
                                      args.lookahead))
                win_names.append(f'{s}:{st}')
        with open(out_dir / f'{split}.pkl', 'wb') as f:
            pickle.dump({'names': win_names, 'ids': ids}, f)
        lens = np.array([len(x) for x in ids])
        print(f'{split}: {len(ids)} windows from {len(names)} songs, {lens.sum():,} tokens, '
              f'len mean {lens.mean():.0f} p95 {np.percentile(lens, 95):.0f} max {lens.max()}')
    json.dump({'lookahead': args.lookahead, 'window': args.window, 'stride': args.stride, 'chords': args.chords},
              open(out_dir / 'meta.json', 'w'), indent=1)
    print('vocab size:', len(tokenizer))


if __name__ == '__main__':
    main()
