"""Bonus task: predict one chord per bar from the 16-bar melody, to feed the chord-planning model.

The chord-planning accompaniment model plays the given chord well (CA 0.56-0.65 with ground-truth
chords) but picks chords poorly when it has to choose them itself. This module replaces that choice
with a dedicated, bidirectional model that sees the whole melody prompt.

Per-bar input: duration-weighted pitch-class chroma of the melody in 4 equal parts of the bar (4 x 12).
Labels: the bar's dominant chord, as in tokenize_bonus.py / eval_bonus.py (24 triads + CHN).

Usage:
  python chord_predictor.py prep    --out ../data/chord_pred/data.pkl
  python chord_predictor.py rule    --data ../data/chord_pred/data.pkl          # template-matching baseline
  python chord_predictor.py train   --data ../data/chord_pred/data.pkl --out_dir ../runs/chord_pred
  python chord_predictor.py predict --ckpt ../runs/chord_pred [more runs ...] --prompt_dir ../POP909/prompts/bonus \
      --out predicted_chords.json
"""
import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
from symusic import Score
from torch import nn

from bonus_utils import CHORDS, bar_chords, bar_lengths_of
from data_process.fix_timesig import fix_one, load_beats

SEG = 4
N_CLASSES = len(CHORDS)  # 0-11 major, 12-23 minor, 24 = N
TEMPLATES = np.zeros((24, 12))
for r in range(12):
    TEMPLATES[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1
    TEMPLATES[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1


def melody_features(score, lengths, track_name='MELODY'):
    """(n_bars, SEG, 12): fraction of each bar part covered by each melody pitch class."""
    tpq = score.ticks_per_quarter
    bar_starts = np.concatenate([[0], np.cumsum(lengths)]) * tpq
    edges = np.concatenate([np.linspace(s, e, SEG + 1)[:-1] for s, e in zip(bar_starts[:-1], bar_starts[1:])]
                           + [[bar_starts[-1]]])
    feats = np.zeros((len(edges) - 1, 12))
    track = next((t for t in score.tracks if t.name.strip().upper() == track_name), None)
    for n in (track.notes if track is not None else []):
        s, e = n.time, n.time + n.duration
        i = max(0, np.searchsorted(edges, s, side='right') - 1)
        while i < len(feats) and edges[i] < e:
            feats[i, n.pitch % 12] += max(0, min(e, edges[i + 1]) - max(s, edges[i]))
            i += 1
    feats /= np.diff(edges)[:, None]
    return feats.reshape(len(lengths), SEG, 12).astype(np.float32)


def labels_of(names):
    return np.array([CHORDS.index(c.removesuffix('_None')) for c in names], dtype=np.int64)


def prompt_inputs(folder):
    """Features, bar lengths and ground-truth labels of one TA bonus prompt (same alignment as generate_bonus)."""
    beat_times, downbeats = load_beats(folder / 'beat_midi.txt')
    with tempfile.TemporaryDirectory() as tmp:
        fixed = Path(tmp) / 'melody.mid'
        _, lengths, origin = fix_one(folder / 'melody.mid', beat_times, downbeats, fixed)
        score = Score(str(fixed))
    starts = origin + np.concatenate([[0], np.cumsum(lengths)])
    lengths = lengths[:int((starts[:-1] < len(beat_times) - 1).sum())]
    labels = None  # ground truth is optional: only used to report accuracy
    if (folder / 'chord_midi.txt').exists():
        labels = labels_of(bar_chords(folder / 'chord_midi.txt', beat_times, downbeats, lengths))
    return melody_features(score, lengths), labels


def rule_predict(feats):
    """Template matching on the bar's melody chroma; empty bar -> N."""
    chroma = feats.sum(1)
    norm = np.linalg.norm(chroma, axis=1)
    pred = (chroma @ TEMPLATES.T / np.maximum(norm, 1e-9)[:, None]).argmax(1)
    return np.where(norm > 0, pred, 24)


class ChordModel(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_bars=64):
        super().__init__()
        self.inp = nn.Linear(SEG * 12, d)
        self.pos = nn.Embedding(max_bars, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLASSES))

    def forward(self, x, pad=None):  # x: (B, T, SEG, 12)
        h = self.inp(x.flatten(2)) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.out(self.enc(h, src_key_padding_mask=pad))


def transpose(feats, labels, k):
    """Shift features and chord labels up by k semitones (N stays N)."""
    lab = labels.copy()
    m = lab < 24
    lab[m] = (lab[m] % 12 + k) % 12 + 12 * (lab[m] // 12)
    return np.roll(feats, k, axis=-1), lab


def windows(songs, win):
    """Non-overlapping windows covering every bar (for evaluation)."""
    out = []
    for f, l in songs:
        for s in range(0, len(l), win):
            out.append((f[s:s + win], l[s:s + win]))
    return out


def collate(items, device):
    T = max(len(l) for _, l in items)
    x = np.zeros((len(items), T, SEG, 12), np.float32)
    y = np.full((len(items), T), -100, np.int64)
    for i, (f, l) in enumerate(items):
        x[i, :len(l)], y[i, :len(l)] = f, l
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    return x, y, y == -100


@torch.no_grad()
def accuracy(model, items, device):
    model.eval()
    x, y, pad = collate(items, device)
    pred = model(x, pad).argmax(-1)
    model.train()
    return ((pred == y) & ~pad).sum().item() / (~pad).sum().item()


def cmd_prep(args):
    splits = json.load(open(args.split))
    data = {}
    for split, names in splits.items():
        songs = []
        for s in names:
            score = Score(str(Path(args.src) / s / f'{s}.mid'))
            lengths = bar_lengths_of(score)
            raw = Path(args.raw_src) / s
            beat_times, downbeats = load_beats(raw / 'beat_midi.txt')
            labels = labels_of(bar_chords(raw / 'chord_midi.txt', beat_times, downbeats, lengths))
            songs.append((melody_features(score, lengths), labels))
        data[split] = songs
        print(f'{split}: {len(songs)} songs, {sum(len(l) for _, l in songs)} bars')
    data['prompts'] = {p.name: prompt_inputs(p) for p in sorted(Path(args.prompt_dir).iterdir()) if p.is_dir()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, open(args.out, 'wb'))


def cmd_rule(args):
    data = pickle.load(open(args.data, 'rb'))
    for split in ['val']:
        f = np.concatenate([f for f, _ in data[split]])
        l = np.concatenate([l for _, l in data[split]])
        print(f'rule baseline {split}: {(rule_predict(f) == l).mean():.3f} ({len(l)} bars)')
    f = np.concatenate([f for f, _ in data['prompts'].values()])
    l = np.concatenate([l for _, l in data['prompts'].values()])
    print(f'rule baseline prompts: {(rule_predict(f) == l).mean():.3f} ({len(l)} bars)')


def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pickle.load(open(args.data, 'rb'))
    train, val = data['train'], windows(data['val'], args.window)
    prompts = list(data['prompts'].values())
    train_eval = windows(train[::10], args.window)  # fixed subset, to watch under/overfitting
    weights = np.array([max(1, len(l) - args.window + 1) for _, l in train], float)
    weights /= weights.sum()

    model = ChordModel(d=args.d, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], -1
    for step in range(1, args.steps + 1):
        batch = []
        for i in rng.choice(len(train), args.batch_size, p=weights):
            f, l = train[i]
            s = rng.integers(0, max(1, len(l) - args.window + 1))
            batch.append(transpose(f[s:s + args.window], l[s:s + args.window], int(rng.integers(12))))
        x, y, pad = collate(batch, device)
        loss = nn.functional.cross_entropy(model(x, pad).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            va, pa = accuracy(model, val, device), accuracy(model, prompts, device)
            log.append({'step': step, 'train_loss': round(loss.item(), 4),
                        'train_acc': round(accuracy(model, train_eval, device), 4), 'val_acc': round(va, 4),
                        'prompt_acc': round(pa, 4)})
            print(log[-1], flush=True)
            json.dump(log, open(out / 'log.json', 'w'), indent=1)
            if va > best:  # model selection on the validation songs only
                best = va
                torch.save({'state': model.state_dict(), 'args': vars(args), 'step': step}, out / 'model.pt')
    print(f'best val acc {best:.4f}')


def cmd_predict(args):
    """Argmax of the probabilities averaged over one or more checkpoints (an ensemble when several)."""
    models = []
    for ckpt in args.ckpt:
        ck = torch.load(Path(ckpt) / 'model.pt', map_location='cpu')
        a = ck['args']
        model = ChordModel(d=a['d'], layers=a['layers'], dropout=a['dropout'])
        model.load_state_dict(ck['state'])
        models.append(model.eval())
    result, hits, n = {}, 0, 0
    for folder in sorted(p for p in Path(args.prompt_dir).iterdir() if p.is_dir()):
        feats, labels = prompt_inputs(folder)
        with torch.no_grad():
            x = torch.from_numpy(feats)[None]
            pred = sum(torch.softmax(m(x), -1) for m in models).argmax(-1)[0].numpy()
        result[folder.name] = [f'{CHORDS[c]}_None' for c in pred]
        if labels is not None:
            hits, n = hits + int((pred == labels).sum()), n + len(labels)
    json.dump(result, open(args.out, 'w'), indent=1)
    if n:
        print(f'prompt chord accuracy vs ground truth: {hits / n:.3f} ({n} bars)')
    print(f'predicted chords -> {args.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--src', default='../data/POP909_fixed/training')
    p.add_argument('--raw_src', default='../POP909/training')
    p.add_argument('--split', default='../data/tokens_basic/split.json')
    p.add_argument('--prompt_dir', default='../POP909/prompts/bonus')
    p.add_argument('--out', required=True)
    p = sub.add_parser('rule')
    p.add_argument('--data', required=True)
    p = sub.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--d', type=int, default=192)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--eval_every', type=int, default=250)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('predict')
    p.add_argument('--ckpt', required=True, nargs='+', help='one or more run folders; several = ensemble')
    p.add_argument('--prompt_dir', required=True)
    p.add_argument('--out', required=True)
    args = ap.parse_args()
    {'prep': cmd_prep, 'rule': cmd_rule, 'train': cmd_train, 'predict': cmd_predict}[args.cmd](args)


if __name__ == '__main__':
    main()
