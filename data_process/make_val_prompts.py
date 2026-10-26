"""Build bonus-style 16-bar prompts from the validation songs, in the TA prompt format.

Gives a held-out set (one window per val song) for tuning bonus inference settings without touching the
10 official prompts. Each folder holds melody.mid, piano.mid, beat_midi.txt and chord_midi.txt with times in
seconds starting at 0, like POP909/prompts/bonus. The window is the 16-bar span (starting on a downbeat)
whose bars most often contain melody, earliest on ties.

Usage:
  python data_process/make_val_prompts.py --raw_src ../POP909/training --split ../data/tokens_basic/split.json \
      --out_dir ../data/val_prompts_bonus
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi


def cut_track(inst, t0, t1, name):
    out = pretty_midi.Instrument(program=0, name=name)
    for n in inst.notes:
        if t0 <= n.start < t1:
            out.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                              start=n.start - t0, end=min(n.end, t1) - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_src', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--bars', type=int, default=16)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    for s in json.load(open(args.split))['val']:
        src = Path(args.raw_src) / s
        beats = np.loadtxt(src / 'beat_midi.txt')
        down = np.flatnonzero(np.isclose(beats[:, 2], 1.0))
        midi = pretty_midi.PrettyMIDI(str(src / f'{s}.mid'))
        tracks = {i.name.strip().upper(): i for i in midi.instruments}
        onsets = np.array([n.start for n in tracks['MELODY'].notes])
        times = beats[down, 0]
        has_mel = np.array([((onsets >= a) & (onsets < b)).any() for a, b in zip(times[:-1], times[1:])])
        n_win = len(down) - args.bars
        if n_win < 1:
            print(f'{s}: too short, skipped')
            continue
        cover = np.array([has_mel[i:i + args.bars].sum() for i in range(n_win)])
        i = int(cover.argmax())
        t0, t1 = times[i], times[i + args.bars]

        out = out_root / s
        out.mkdir(parents=True, exist_ok=True)
        for file_name, track in [('melody.mid', 'MELODY'), ('piano.mid', 'PIANO')]:
            pm = pretty_midi.PrettyMIDI()
            pm.instruments.append(cut_track(tracks[track], t0, t1, track))
            pm.write(str(out / file_name))
        sel = beats[down[i]:down[i + args.bars] + 1]
        with open(out / 'beat_midi.txt', 'w') as f:
            for t, a, b in sel:
                f.write(f'{t - t0:.6f} {a:.1f} {b:.1f}\n')
        with open(out / 'chord_midi.txt', 'w') as f:
            for line in open(src / 'chord_midi.txt'):
                p = line.split()
                a, b = max(float(p[0]), t0), min(float(p[1]), t1)
                if len(p) >= 3 and b > a:
                    f.write(f'{a - t0:.6f} {b - t0:.6f} {p[2]}\n')
        print(f'{s}: bars {i}-{i + args.bars - 1}, {cover[i]}/{args.bars} bars with melody')


if __name__ == '__main__':
    main()
