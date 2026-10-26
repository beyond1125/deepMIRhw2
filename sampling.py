"""Shared inference utilities: load a checkpoint, sample REMI tokens bar by bar, write MIDI."""
import json
from pathlib import Path

import numpy as np
import torch
from miditok import REMI
from symusic import Score, Track
from transformers import AutoModelForCausalLM

CONFIG_FILE = Path(__file__).parent / 'configs' / 'sampling.json'


def pick_device(device='auto'):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def load(ckpt, device):
    tokenizer = REMI(params=Path(ckpt) / 'tokenizer_remi.json')
    model = AutoModelForCausalLM.from_pretrained(ckpt, attn_implementation='sdpa').to(device).eval()
    return model, tokenizer


def sampling_config(name=None, **overrides):
    """Named preset from configs/sampling.json, with non-None overrides applied."""
    cfg = {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0}
    if name:
        cfg.update(json.load(open(CONFIG_FILE))[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def filter_logits(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p  # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_bars(model, tokenizer, prefixes, n_bars, cfg, max_len=None, generator=None):
    """Continue each prefix (list of token ids, starting with BOS) until it holds n_bars bars.

    EOS/BOS/PAD are masked so every piece reaches n_bars; a sequence stops right before the
    Bar token that would open bar n_bars + 1. Returns a list of token-id lists (BOS stripped)
    and a list of flags telling whether the context limit was hit first.
    """
    device = next(model.parameters()).device
    max_len = max_len or model.config.max_position_embeddings
    bar = tokenizer['Bar_None']
    banned = [tokenizer['BOS_None'], tokenizer['EOS_None'], tokenizer['PAD_None']]

    seqs = [list(p) for p in prefixes]
    bars = [sum(t == bar for t in s) for s in seqs]
    done = [b > n_bars for b in bars]
    truncated = [False] * len(seqs)

    # left-pad prefixes of different lengths into one batch
    L = max(len(s) for s in seqs)
    pad = tokenizer['PAD_None']
    x = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=device)
    attn = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)

    out = model(input_ids=x, attention_mask=attn, position_ids=pos, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    while not all(done):
        logits = logits.float()
        logits[:, banned] = float('-inf')
        probs = torch.softmax(filter_logits(logits, **cfg), dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()
        for i, t in enumerate(nxt):
            if done[i]:
                continue
            if t == bar and bars[i] == n_bars:
                done[i] = True
                continue
            seqs[i].append(t)
            bars[i] += t == bar
            if len(seqs[i]) >= max_len:
                done[i] = truncated[i] = True
        attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=-1)
        pos = pos[:, -1:] + 1
        out = model(input_ids=torch.tensor(nxt, device=device)[:, None], attention_mask=attn,
                    position_ids=pos, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    return [s[1:] for s in seqs], truncated


def generate_complete(model, tokenizer, prefixes, n_bars, cfg, generator=None, max_retries=10):
    """generate_bars, re-sampling any piece that hits the context limit before reaching n_bars.

    Returns token-id lists and, per piece, how many failed attempts were discarded.
    """
    bar = tokenizer['Bar_None']
    results, retries = [None] * len(prefixes), [0] * len(prefixes)
    todo = list(range(len(prefixes)))
    for attempt in range(max_retries + 1):
        seqs, trunc = generate_bars(model, tokenizer, [prefixes[i] for i in todo], n_bars, cfg, generator=generator)
        failed = []
        for i, ids, tr in zip(todo, seqs, trunc):
            if tr and ids.count(bar) < n_bars and attempt < max_retries:
                retries[i] += 1
                failed.append(i)
            else:
                results[i] = ids
        if not failed:
            break
        todo = failed
    return results, retries


def merge_tracks(score):
    """Collapse all tracks into one piano track (same as training preprocessing)."""
    merged = Track(name='PIANO', program=0)
    for track in score.tracks:
        merged.notes.extend(track.notes)
    merged.notes.sort(key=lambda n: (n.time, n.pitch))
    score.tracks.clear()
    score.tracks.append(merged)
    return score


def encode_score(tokenizer, score):
    seq = tokenizer.encode(merge_tracks(score))
    seq = seq[0] if isinstance(seq, list) else seq
    return [tokenizer['BOS_None']] + list(seq.ids)


def save_midi(tokenizer, ids, path):
    score = tokenizer.decode([ids])
    for track in score.tracks:
        track.name, track.program = 'PIANO', 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(path))
    return score
