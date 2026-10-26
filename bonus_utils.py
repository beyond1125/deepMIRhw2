"""Bonus task helpers: per-bar REMI segments for MELODY / PIANO and the lookahead interleaving.

Sequence layout with lookahead k (melody is shown k bars ahead of the accompaniment):
  BOS  <MEL> m_1 ... <MEL> m_k  [<MEL> m_{j+k}] <ACC> a_j  for j = 1..N
Each m_j / a_j is that bar's REMI tokens and starts with Bar + TimeSig.
With chord planning, every a_j is Bar TimeSig CH<c> ..., where c is the bar's dominant chord
(24 major/minor triads, CHN = none), so the model first predicts the harmony, then the notes.
"""
from collections import Counter

import numpy as np
from miditok import REMI, TokenizerConfig
from symusic import Score, Track

MEL, ACC = 'MEL_None', 'ACC_None'


CHORDS = [f'CH{i}' for i in range(24)] + ['CHN']  # 0-11 major roots, 12-23 minor roots
ROOTS = {'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
         'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
         'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}


def build_tokenizer(chords=False):
    """Same REMI settings as the Basic task, plus the two segment tokens (and chord tokens)."""
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
        special_tokens=['PAD', 'BOS', 'EOS', 'MEL', 'ACC'] + (CHORDS if chords else []),
    )
    return REMI(config)


def bar_lengths_of(score):
    """Beats per bar (n/4 time signatures) covering the whole score."""
    tpq = score.ticks_per_quarter
    end = max((t.end() for t in score.tracks), default=0)
    ts = sorted(score.time_signatures, key=lambda t: t.time)
    lengths, tick, i = [], 0, 0
    while tick < end or not lengths:
        while i + 1 < len(ts) and ts[i + 1].time <= tick:
            i += 1
        lengths.append(ts[i].numerator)
        tick += ts[i].numerator * tpq
    return lengths


def track_bars(tokenizer, score, track_name, lengths):
    """REMI ids of one track split per bar; padded with empty bars to len(lengths)."""
    single = Score(score.ticks_per_quarter)
    single.tempos = score.tempos
    single.time_signatures = score.time_signatures
    track = next((t for t in score.tracks if t.name.strip().upper() == track_name), None)
    single.tracks.append(track.copy() if track is not None else Track(name=track_name, program=0))
    bar = tokenizer['Bar_None']

    ids = []
    if single.note_num() > 0:
        seq = tokenizer.encode(single)
        ids = list((seq[0] if isinstance(seq, list) else seq).ids)
    starts = [i for i, t in enumerate(ids) if t == bar]
    bars = [ids[s:e] for s, e in zip(starts, starts[1:] + [len(ids)])]
    for j in range(len(bars), len(lengths)):
        bars.append(empty_bar(tokenizer, lengths[j]))
    return bars[:len(lengths)]


def empty_bar(tokenizer, n_beats):
    return [tokenizer['Bar_None'], tokenizer[f'TimeSig_{n_beats}/4']]


def interleave(tokenizer, mel_bars, acc_bars, k):
    """Full training sequence for one window (see module docstring)."""
    mel, acc = tokenizer[MEL], tokenizer[ACC]
    n = len(acc_bars)
    seq = [tokenizer['BOS_None']]
    for j in range(min(k, n)):
        seq += [mel] + mel_bars[j]
    for j in range(n):
        if j + k < n:
            seq += [mel] + mel_bars[j + k]
        seq += [acc] + acc_bars[j]
    return np.array(seq, dtype=np.int16)


def beats_to_seconds(beats, beat_times):
    """Inverse of the beat-grid mapping in fix_timesig, with linear extrapolation."""
    idx = np.arange(len(beat_times))
    t = np.interp(beats, idx, beat_times)
    first, last = beat_times[1] - beat_times[0], beat_times[-1] - beat_times[-2]
    t = np.where(beats < 0, beat_times[0] + beats * first, t)
    t = np.where(beats > idx[-1], beat_times[-1] + (beats - idx[-1]) * last, t)
    return t


def chord_class(label):
    """POP909 chord label -> 0..23 (major / minor triad), or None for N, dim, aug, sus.

    Same triad mapping as the TA's eval_bonus.py (parse_pop909_chord).
    """
    if label == 'N':
        return None
    root, _, quality = label.partition(':')
    if root not in ROOTS:
        return None
    quality = (quality or 'maj').split('/')[0].split('(')[0]
    if quality.startswith(('dim', 'hdim', 'aug')) or 'sus' in quality:
        return None
    return ROOTS[root] + (12 if quality.startswith('min') else 0)


def bar_chords(chord_file, beat_times, downbeats, lengths):
    """Dominant chord token name (by overlap) for each bar of the beat-aligned score."""
    from data_process.fix_timesig import bar_lengths
    dominant = Counter(np.diff(downbeats).astype(int).tolist()).most_common(1)[0][0] if len(downbeats) > 1 else 4
    origin, _ = bar_lengths(downbeats, len(beat_times), dominant)
    edges = beats_to_seconds(origin + np.concatenate([[0], np.cumsum(lengths)]).astype(float), beat_times)
    rows = []
    for line in open(chord_file):
        p = line.split()
        if len(p) >= 3:
            rows.append((float(p[0]), float(p[1]), chord_class(p[2])))
    names = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        votes = Counter()
        for s, e, c in rows:
            ov = min(e, t1) - max(s, t0)
            if ov > 0:
                votes[c] += ov
        best = votes.most_common(1)[0][0] if votes else None
        names.append('CHN_None' if best is None else f'CH{best}_None')
    return names


def add_chords(tokenizer, acc_bars, chord_names):
    """Insert each bar's chord token right after its Bar + TimeSig tokens."""
    return [b[:2] + [tokenizer[c]] + b[2:] for b, c in zip(acc_bars, chord_names)]
