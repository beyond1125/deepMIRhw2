"""Re-align POP909 MIDI files to the beat grid in beat_midi.txt and write correct time signatures.

POP909 .mid files mostly carry wrong time signatures (e.g. 2/4, 1/4), and the TA prompts are
stored at a dummy 120 BPM, so their ticks do not line up with bars. For every song folder we:
  1. map each note's onset/offset (seconds) to a fractional beat index via beat_midi.txt
     (column 1 = beat time, column 3 = downbeat flag), extrapolating linearly at both ends;
  2. shift so that bar 1 starts on tick 0 (pickup beats are padded into a full leading bar);
  3. derive one time signature per bar from the gap between consecutive downbeats (n/4);
  4. write a single tempo (median beat interval) and save at 480 ticks per beat.
fix_one returns (dominant bar length, beats per bar, origin), where origin is the beat index in
beat_midi.txt that tick 0 corresponds to (negative when a pickup was padded).

Usage:
  python fix_timesig.py --src ../../POP909 --dst ../../data/POP909_fixed
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile
from miditoolkit.midi.containers import Instrument, Note, TempoChange, TimeSignature

TPB = 480
MAX_NUM = 8  # bars longer than this are split into chunks of the song's dominant length


def load_beats(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], np.where(a[:, 2] == 1)[0]


def beat_position(t, beat_times):
    """Fractional beat index for times t (seconds), linear extrapolation outside the grid."""
    b = np.interp(t, beat_times, np.arange(len(beat_times)))
    first, last = beat_times[1] - beat_times[0], beat_times[-1] - beat_times[-2]
    b = np.where(t < beat_times[0], (t - beat_times[0]) / first, b)
    b = np.where(t > beat_times[-1], len(beat_times) - 1 + (t - beat_times[-1]) / last, b)
    return b


def bar_lengths(downbeats, n_beats_total, dominant):
    """Beats per bar from bar 1 (padded pickup) to the end; returns (origin_beat, [lengths])."""
    d0 = int(downbeats[0])
    n_pick = math.ceil(d0 / dominant)
    origin = d0 - n_pick * dominant
    lengths = [dominant] * n_pick
    for g in np.diff(downbeats).astype(int):
        while g > MAX_NUM:
            lengths.append(dominant)
            g -= dominant
        if g > 0:
            lengths.append(int(g))
    tail = n_beats_total - int(downbeats[-1])
    lengths += [dominant] * max(1, math.ceil(tail / dominant))
    return origin, lengths


def fix_one(mid_path, beat_times, downbeats, out_path):
    midi = MidiFile(str(mid_path))
    tick2sec = midi.get_tick_to_time_mapping()
    dominant = Counter(np.diff(downbeats).astype(int).tolist()).most_common(1)[0][0] if len(downbeats) > 1 else 4

    notes = [(inst, n) for inst in midi.instruments for n in inst.notes]
    sec = lambda ticks: tick2sec[np.minimum(ticks, len(tick2sec) - 1)]
    starts = beat_position(sec([n.start for _, n in notes]), beat_times)
    ends = beat_position(sec([n.end for _, n in notes]), beat_times)

    n_beats_total = int(math.ceil(max(ends.max(), len(beat_times) - 1)))
    origin, lengths = bar_lengths(downbeats, n_beats_total, dominant)

    out = MidiFile(ticks_per_beat=TPB)
    bpm = 60.0 / float(np.median(np.diff(beat_times)))
    out.tempo_changes = [TempoChange(bpm, 0)]
    tick, prev = 0, None
    for L in lengths:
        if L != prev:
            out.time_signature_changes.append(TimeSignature(L, 4, tick))
            prev = L
        tick += L * TPB

    new_insts = {}
    for (inst, n), s, e in zip(notes, starts, ends):
        st = max(0, int(round((s - origin) * TPB)))
        en = max(st + 1, int(round((e - origin) * TPB)))
        key = id(inst)
        if key not in new_insts:
            new_insts[key] = Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        new_insts[key].notes.append(Note(velocity=n.velocity, pitch=n.pitch, start=st, end=en))
    for inst in midi.instruments:  # keep original track order
        if id(inst) in new_insts:
            new_insts[id(inst)].notes.sort(key=lambda n: (n.start, n.pitch))
            out.instruments.append(new_insts[id(inst)])

    out.max_tick = max(tick, max(n.end for i in out.instruments for n in i.notes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.dump(str(out_path))
    return dominant, lengths, origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='TA POP909 folder (contains training/ and prompts/)')
    ap.add_argument('--dst', required=True, help='output folder, same layout as --src')
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    stats = Counter()
    for beat_file in sorted(src.rglob('beat_midi.txt')):
        beat_times, downbeats = load_beats(beat_file)
        for mid in sorted(beat_file.parent.glob('*.mid')):
            dominant, lengths, _ = fix_one(mid, beat_times, downbeats, dst / mid.relative_to(src))
            stats[f'dominant {dominant}/4'] += 1
            stats.update(f'bar {L}/4' for L in lengths)
        stats['folders'] += 1
    for k, v in sorted(stats.items()):
        print(f'{k:>16}: {v}')


if __name__ == '__main__':
    main()
