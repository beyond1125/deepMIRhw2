"""Train a decoder-only Transformer (Llama architecture, from scratch) on REMI token sequences.

Training samples are random windows that start on a Bar token, prefixed with BOS, with random
pitch transposition. Validation uses fixed, non-augmented windows; the best checkpoint by val
loss is kept in <out_dir>/best.

Usage:
  python train.py --data_dir ../data/tokens_basic --tokenizer configs/tokenizer_remi.json \
      --out_dir checkpoints/basic
"""
import argparse
import csv
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from miditok import REMI
from transformers import AutoModelForCausalLM, LlamaConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out_dir', required=True)
    # model
    ap.add_argument('--n_layer', type=int, default=8)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--d_ff', type=int, default=2048)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--dropout', type=float, default=0.0)
    # optimization
    ap.add_argument('--batch_size', type=int, default=16, help='0 = largest that fits in GPU memory')
    ap.add_argument('--train_tokens', type=float, default=0, help='if > 0, max_steps = train_tokens / (batch_size * ctx)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--max_steps', type=int, default=6000)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--patience', type=int, default=6, help='evals without val improvement before stopping')
    ap.add_argument('--transpose', type=int, default=6, help='max semitone shift (both directions)')
    ap.add_argument('--bos_windows', action='store_true',
                    help='sequences are self-contained windows (bonus task): always start at BOS, never mid-sequence')
    ap.add_argument('--diverge_tol', type=float, default=0.5, help='stop if val loss exceeds best by this much')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


class WindowSampler:
    """Random bar-aligned, fully packed windows with BOS prefix and pitch transposition."""

    def __init__(self, seqs, tokenizer, ctx, transpose, rng, bos_windows=False):
        self.seqs = [s.astype(np.int64) for s in seqs]
        self.ctx, self.transpose, self.rng, self.bos_windows = ctx, transpose, rng, bos_windows
        self.bos, self.pad = tokenizer['BOS_None'], tokenizer['PAD_None']
        bar = tokenizer['Bar_None']
        # window starts: bars leaving at least half a context, to limit padding
        self.bars = []
        for s in self.seqs:
            b = np.where(s == bar)[0]
            far = b[len(s) - b >= ctx // 2]
            self.bars.append(far if len(far) else b[:1])
        lens = np.array([len(s) for s in self.seqs], dtype=np.float64)
        self.probs = lens / lens.sum()

        vocab = tokenizer.vocab
        pitch_of = {i: int(t.split('_')[1]) for t, i in vocab.items() if t.startswith('Pitch_')}
        id_of = {p: i for i, p in pitch_of.items()}
        self.pitch_ids = np.array(sorted(pitch_of))
        self.pitch_of = np.full(len(vocab), -1)
        for i, p in pitch_of.items():
            self.pitch_of[i] = p
        self.pmin, self.pmax = min(id_of), max(id_of)
        # chord tokens (bonus chord planning): CH0-11 major, CH12-23 minor; roots move with the shift
        chord_id = {int(t[2:-5]): i for t, i in vocab.items() if t.startswith('CH') and t[2:-5].isdigit()}
        self.luts = {}
        for k in range(-transpose, transpose + 1):
            lut = np.arange(len(vocab))
            for i, p in pitch_of.items():
                if p + k in id_of:
                    lut[i] = id_of[p + k]
            for c, i in chord_id.items():
                lut[i] = chord_id[(c // 12) * 12 + (c % 12 + k) % 12]
            self.luts[k] = lut

    def sample(self):
        i = self.rng.choice(len(self.seqs), p=self.probs)
        seq, bars = self.seqs[i], self.bars[i]
        start = 0 if self.bos_windows or self.rng.random() < 0.1 else int(self.rng.choice(bars))
        win = seq[start:start + self.ctx - (start > 0)]
        if start > 0:
            win = np.concatenate([[self.bos], win])
        # pack: fill the rest of the context with other songs from their beginning (BOS)
        while len(win) < self.ctx:
            other = self.seqs[self.rng.choice(len(self.seqs), p=self.probs)]
            win = np.concatenate([win, other[:self.ctx - len(win)]])
        if self.transpose:
            p = self.pitch_of[win]
            p = p[p >= 0]
            if len(p):
                lo = max(-self.transpose, self.pmin - p.min())
                hi = min(self.transpose, self.pmax - p.max())
                win = self.luts[int(self.rng.integers(lo, hi + 1))][win]
        return win

    def batch(self, n):
        return collate([self.sample() for _ in range(n)], self.ctx, self.pad)


def collate(wins, ctx, pad):
    x = np.full((len(wins), ctx), pad, dtype=np.int64)
    y = np.full((len(wins), ctx), -100, dtype=np.int64)
    for j, w in enumerate(wins):
        x[j, :len(w)] = w
        y[j, :len(w)] = w
    return torch.from_numpy(x), torch.from_numpy(y)


def val_windows(seqs, ctx, bos):
    """Fixed, non-overlapping windows covering every validation song."""
    wins = []
    for s in seqs:
        s = s.astype(np.int64)
        wins.append(s[:ctx])
        for st in range(ctx, len(s), ctx - 1):
            wins.append(np.concatenate([[bos], s[st:st + ctx - 1]]))
    return wins


@torch.no_grad()
def evaluate(model, wins, ctx, pad, bs, device):
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(wins), bs):
        x, y = collate(wins[k:k + bs], ctx, pad)
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=(y != -100).long()).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1),
            ignore_index=-100, reduction='sum')
        tot += loss.item()
        n += (y[:, 1:] != -100).sum().item()
    model.train()
    return tot / n


def probe_batch_size(model, ctx, vocab, device, frac=0.85):
    """Largest batch size whose peak memory for one full-length step stays under frac of the GPU."""
    total = torch.cuda.get_device_properties(device).total_memory
    for bs in (128, 96, 80, 64, 56, 48, 40, 32, 24, 16, 8):
        try:
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randint(0, vocab, (bs, ctx), device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(input_ids=x, labels=x).loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            ok = peak < frac * total
        except torch.OutOfMemoryError:
            peak, ok = float('nan'), False
        model.zero_grad(set_to_none=True)
        x = None
        torch.cuda.empty_cache()
        print(f'probe bs {bs}: peak {peak / 2**30:.1f} GiB {"ok" if ok else "too big"}', flush=True)
        if ok:
            return bs
    raise RuntimeError('no batch size fits')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = REMI(params=args.tokenizer)
    with open(Path(args.data_dir) / 'train.pkl', 'rb') as f:
        train = pickle.load(f)['ids']
    with open(Path(args.data_dir) / 'val.pkl', 'rb') as f:
        val = pickle.load(f)['ids']
    sampler = WindowSampler(train, tokenizer, args.ctx, args.transpose, rng, args.bos_windows)
    vwins = val_windows(val, args.ctx, tokenizer['BOS_None'])
    pad = tokenizer['PAD_None']

    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=args.d_model, intermediate_size=args.d_ff,
        num_hidden_layers=args.n_layer, num_attention_heads=args.n_head,
        num_key_value_heads=args.n_head, max_position_embeddings=args.ctx,
        attention_dropout=args.dropout, rope_theta=10000.0,
        bos_token_id=tokenizer['BOS_None'], eos_token_id=tokenizer['EOS_None'], pad_token_id=pad,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa').to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params / 1e6:.1f}M, train windows/epoch ~{sum(len(s) for s in train) / args.ctx:.0f}, '
          f'val windows {len(vwins)}', flush=True)

    if args.batch_size == 0:
        args.batch_size = probe_batch_size(model, args.ctx, len(tokenizer), device)
    if args.train_tokens > 0:
        args.max_steps = int(args.train_tokens / (args.batch_size * args.ctx))
    print(f'batch size {args.batch_size}, max steps {args.max_steps}', flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / args.warmup) * \
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.max_steps))))
    step_fn = torch.compile(model) if args.compile else model

    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args) | {'n_params': n_params}, f, indent=1)
    log = open(out_dir / 'log.csv', 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['step', 'train_loss', 'val_loss', 'lr', 'tok_per_s', 'max_grad_norm'])

    best, bad, run_loss, run_n, t0, toks, gmax = float('inf'), 0, 0.0, 0, time.time(), 0, 0.0
    for step in range(1, args.max_steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        x, y = sampler.batch(args.batch_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = step_fn(input_ids=x, labels=y)  # windows are fully packed: no padding mask needed
        out.loss.backward()
        gmax = max(gmax, torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        run_loss += out.loss.item()
        run_n += 1
        toks += (y != -100).sum().item()

        if step % args.eval_every == 0 or step == args.max_steps:
            vl = evaluate(model, vwins, args.ctx, pad, args.batch_size, device)
            tps = toks / (time.time() - t0)
            writer.writerow([step, f'{run_loss / run_n:.4f}', f'{vl:.4f}', f'{lr_at(step):.2e}', f'{tps:.0f}', f'{gmax:.2f}'])
            log.flush()
            print(f'step {step} train {run_loss / run_n:.4f} val {vl:.4f} lr {lr_at(step):.2e} '
                  f'gnorm max {gmax:.2f} {tps:,.0f} tok/s {(time.time() - t0) / 60:.1f} min', flush=True)
            run_loss, run_n, gmax = 0.0, 0, 0.0
            if vl < best:
                best, bad = vl, 0
                model.save_pretrained(out_dir / 'best')
                shutil.copy(args.tokenizer, out_dir / 'best' / 'tokenizer_remi.json')
                if (Path(args.data_dir) / 'meta.json').exists():  # bonus: lookahead setting
                    shutil.copy(Path(args.data_dir) / 'meta.json', out_dir / 'best' / 'data_meta.json')
            else:
                bad += 1
                if vl > best + args.diverge_tol:
                    print(f'diverged at step {step} (val {vl:.4f} > best {best:.4f} + {args.diverge_tol}), stopping', flush=True)
                    break
                if bad >= args.patience:
                    print(f'early stop at step {step}, best val {best:.4f}', flush=True)
                    break
    log.close()
    print(f'done, best val {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
