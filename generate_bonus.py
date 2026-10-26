"""Bonus task: generate the PIANO accompaniment for each 16-bar TA melody prompt.

Bar by bar, the given melody bars are inserted (k bars ahead, k read from the checkpoint) and the
accompaniment bar is sampled until the model emits the next segment token. The output MIDI keeps
the original MELODY notes and puts the generated PIANO on the prompt's original timeline
(seconds from beat_midi.txt), which is what eval_bonus.py expects. Files are named <prompt id>.mid.

Usage:
  python generate_bonus.py --ckpt checkpoints/bonus_k1 --prompt_dir POP909/prompts/bonus --output_dir out/bonus
  python generate_bonus.py --oracle --prompt_dir POP909/prompts/bonus --output_dir out/oracle   # GT piano round trip
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from symusic import Score

from bonus_utils import ACC, MEL, bar_chords, beats_to_seconds, build_tokenizer, empty_bar, track_bars
from data_process.fix_timesig import fix_one, load_beats
from sampling import filter_logits, load, pick_device, sampling_config


def load_prompt(tokenizer, folder, file_name, track_name):
    """Beat-aligned bars of one prompt track, plus what is needed to map back to seconds."""
    beat_times, downbeats = load_beats(folder / 'beat_midi.txt')
    with tempfile.TemporaryDirectory() as tmp:
        fixed = Path(tmp) / file_name
        _, lengths, origin = fix_one(folder / file_name, beat_times, downbeats, fixed)
        score = Score(str(fixed))
    # bars that start inside the beat file
    starts = origin + np.concatenate([[0], np.cumsum(lengths)])
    n_bars = int((starts[:-1] < len(beat_times) - 1).sum())
    lengths = lengths[:n_bars]
    return track_bars(tokenizer, score, track_name, lengths), lengths, origin, beat_times, downbeats


@torch.no_grad()
def generate_accompaniment(model, tokenizer, mel_bars, lengths, k, cfg, generator, max_bar_tokens=512,
                           chord_temperature=None, forced_chords=None, chord_bias=0.0):
    device = next(model.parameters()).device
    ctx = model.config.max_position_embeddings
    stop = {tokenizer[MEL], tokenizer[ACC], tokenizer['EOS_None']}
    chord_ids = [i for t, i in tokenizer.vocab.items() if t.startswith('CH') and t.endswith('_None')]
    not_chord = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
    not_chord[chord_ids] = False
    banned = [tokenizer['BOS_None'], tokenizer['PAD_None'], tokenizer['Bar_None']] + chord_ids + \
        [i for t, i in tokenizer.vocab.items() if t.startswith('TimeSig_')]
    # chord-aware decoding: pitch tokens outside the bar's triad get their logit lowered by chord_bias
    pitch_pc = {i: int(t.split('_')[1]) % 12 for t, i in tokenizer.vocab.items() if t.startswith('Pitch_')}
    off_chord = {}
    for c in range(24):
        tones = {c % 12, (c % 12 + (3 if c >= 12 else 4)) % 12, (c % 12 + 7) % 12}
        off_chord[f'CH{c}_None'] = torch.tensor([i for i, pc in pitch_pc.items() if pc not in tones], device=device)
    state = {'past': None, 'len': 0}

    def feed(ids):
        out = model(input_ids=torch.tensor([ids], device=device), past_key_values=state['past'], use_cache=True)
        state['past'], state['len'] = out.past_key_values, state['len'] + len(ids)
        return out.logits[0, -1].float()

    n = len(lengths)
    prefix = [tokenizer['BOS_None']]
    for j in range(min(k, n)):
        prefix += [tokenizer[MEL]] + mel_bars[j]
    logits = feed(prefix)

    acc_bars, chords, hit_ctx = [], [], False
    for j in range(n):
        forced = ([tokenizer[MEL]] + mel_bars[j + k] if j + k < n else []) + \
            [tokenizer[ACC]] + empty_bar(tokenizer, lengths[j])
        if hit_ctx or state['len'] + len(forced) >= ctx:
            hit_ctx = True
            acc_bars.append(empty_bar(tokenizer, lengths[j]))
            chords.append(None)
            continue
        logits = feed(forced)
        bar = empty_bar(tokenizer, lengths[j])
        if chord_ids:  # chord planning: the first accompaniment token is the bar's chord
            logits[not_chord] = float('-inf')
            if forced_chords is not None:  # diagnostic: ground-truth chord per bar
                c = tokenizer[forced_chords[j]]
            elif chord_temperature == 0:
                c = int(logits.argmax())
            else:
                ccfg = cfg if chord_temperature is None else dict(cfg, temperature=chord_temperature)
                probs = torch.softmax(filter_logits(logits, **ccfg), dim=-1)
                c = int(torch.multinomial(probs, 1, generator=generator))
            chords.append(tokenizer[c])
            bar.append(c)
            logits = feed([c])
        off = off_chord.get(chords[-1]) if chord_ids and chord_bias else None
        while len(bar) < max_bar_tokens and state['len'] < ctx:
            logits[banned] = float('-inf')
            if off is not None:
                logits[off] -= chord_bias
            probs = torch.softmax(filter_logits(logits, **cfg), dim=-1)
            t = int(torch.multinomial(probs, 1, generator=generator))
            if t in stop:
                break
            bar.append(t)
            logits = feed([t])
        hit_ctx = hit_ctx or state['len'] >= ctx
        acc_bars.append(bar)
    return acc_bars, chords, hit_ctx


def write_output(tokenizer, acc_bars, origin, beat_times, melody_path, out_path):
    """MELODY (original notes) + PIANO (generated, mapped back to the prompt's seconds)."""
    special = {i for t, i in tokenizer.vocab.items() if t.endswith('_None') and not t.startswith('Bar')}
    score = tokenizer.decode([[t for b in acc_bars for t in b if t not in special]])
    tpq = score.ticks_per_quarter
    notes = [n for tr in score.tracks for n in tr.notes]
    starts = beats_to_seconds(np.array([n.time / tpq for n in notes]) + origin, beat_times)
    ends = beats_to_seconds(np.array([(n.time + n.duration) / tpq for n in notes]) + origin, beat_times)

    out = pretty_midi.PrettyMIDI()
    melody = pretty_midi.PrettyMIDI(str(melody_path)).instruments[0]
    melody.name = 'MELODY'
    piano = pretty_midi.Instrument(program=0, name='PIANO')
    for n, s, e in zip(notes, starts, ends):
        piano.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                            start=float(s), end=float(max(e, s + 1e-3))))
    out.instruments += [melody, piano]
    out.write(str(out_path))
    return len(piano.notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', help='bonus checkpoint folder (tokenizer_remi.json, data_meta.json)')
    ap.add_argument('--prompt_dir', required=True, help='TA folder with <id>/melody.mid and beat_midi.txt')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default='A_t1.0_p0.95', help='preset name in configs/sampling.json')
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--top_k', type=int)
    ap.add_argument('--top_p', type=float)
    ap.add_argument('--chord_temperature', type=float, default=None,
                    help='chord-planning models: temperature for chord tokens (0 = greedy); default = same as notes')
    ap.add_argument('--oracle_chords', action='store_true',
                    help='diagnostic only: force the ground-truth chord_midi.txt chords instead of predicting them')
    ap.add_argument('--chord_bias', type=float, default=0.0,
                    help='chord-planning models: subtract this from the logits of pitches outside the bar chord (inf = chord tones only)')
    ap.add_argument('--chord_file', help='JSON {prompt id: [chord token per bar]} to force, e.g. from chord_predictor.py')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--oracle', action='store_true', help='use the ground-truth piano.mid instead of the model')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.oracle:
        tokenizer, model, k = build_tokenizer(), None, None
    else:
        device = pick_device(args.device)
        model, tokenizer = load(args.ckpt, device)
        k = json.load(open(Path(args.ckpt) / 'data_meta.json'))['lookahead']
        cfg = sampling_config(args.config, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        print(f'device {device}, lookahead {k}, sampling {cfg}', flush=True)

    chord_file = json.load(open(args.chord_file)) if args.chord_file else None
    meta = []
    for folder in sorted(p for p in Path(args.prompt_dir).iterdir() if p.is_dir()):
        mel_bars, lengths, origin, beat_times, downbeats = load_prompt(tokenizer, folder, 'melody.mid', 'MELODY')
        if args.oracle:
            acc_bars, chords, hit = load_prompt(tokenizer, folder, 'piano.mid', 'PIANO')[0], [], False
        else:
            forced = bar_chords(folder / 'chord_midi.txt', beat_times, downbeats, lengths) if args.oracle_chords \
                else chord_file[folder.name] if chord_file else None
            acc_bars, chords, hit = generate_accompaniment(model, tokenizer, mel_bars, lengths, k, cfg, gen,
                                                           chord_temperature=args.chord_temperature,
                                                           forced_chords=forced, chord_bias=args.chord_bias)
        n_notes = write_output(tokenizer, acc_bars, origin, beat_times, folder / 'melody.mid',
                               out_dir / f'{folder.name}.mid')
        meta.append({'file': f'{folder.name}.mid', 'bars': len(lengths), 'piano_notes': n_notes,
                     'acc_tokens': sum(len(b) for b in acc_bars), 'hit_context_limit': hit,
                     'predicted_chords': chords})
        print(f'{folder.name}: {len(lengths)} bars, {n_notes} piano notes', flush=True)
    json.dump({'oracle': args.oracle, 'lookahead': k, 'sampling': None if args.oracle else cfg,
               'chord_temperature': args.chord_temperature, 'oracle_chords': args.oracle_chords, 'chord_file': args.chord_file, 'chord_bias': args.chord_bias,
               'seed': args.seed, 'pieces': meta}, open(out_dir / 'generation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
