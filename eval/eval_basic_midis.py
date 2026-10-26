"""Run the TA's Basic-task metrics (H1, H4, GS) on a folder of generated MIDI files.

The TA pipeline (tokenization.py -> eval_basic.py) expects POP909-style song folders with a
beat_midi.txt. For each generated .mid we derive that beat file from the MIDI's own tempo and
time signatures, then call the TA scripts unchanged, so generated pieces and real data go
through exactly the same tokenization.

Usage:
  python eval/eval_basic_midis.py --midi_dir results/task_a/A_t1.0_p0.95 \
      --ta_dir <folder with tokenization.py, eval_basic.py> --musdr_parent <folder containing MusDr/> \
      --output_csv results/task_a/A_t1.0_p0.95/metrics.csv
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi


def write_beat_file(midi_path, out_path):
    """beat_midi.txt rows: <beat time (s)> <strong beat flag> <downbeat flag>, as in POP909."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = pm.get_beats()
    downbeats = set(np.round(pm.get_downbeats(), 6))
    with open(out_path, 'w') as f:
        for i, t in enumerate(beats):
            down = float(round(t, 6) in downbeats)
            f.write(f'{t:.6f} {float(i % 2 == 0)} {down}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--midi_dir', required=True)
    ap.add_argument('--ta_dir', required=True, help='folder with the TA tokenization.py and eval_basic.py')
    ap.add_argument('--musdr_parent', required=True, help='folder that contains the MusDr repository')
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        songs = Path(tmp) / 'songs'
        for mid in sorted(Path(args.midi_dir).glob('*.mid')):
            d = songs / mid.stem
            d.mkdir(parents=True)
            shutil.copy(mid, d / mid.name)
            write_beat_file(mid, d / 'beat_midi.txt')
            (d / 'chord_midi.txt').touch()  # required by tokenization.py, not used by the metrics

        ta = Path(args.ta_dir)
        tok2idx, seqs = Path(tmp) / 'tok2idx.json', Path(tmp) / 'sequences.json'
        subprocess.run([sys.executable, str(ta / 'tokenization.py'), '--dataset_path', str(songs),
                        '--tok2idx_output', str(tok2idx), '--sequences_output', str(seqs),
                        '--pos_per_beat', '16'], check=True, stdout=subprocess.DEVNULL)
        env = dict(os.environ, PYTHONPATH=str(Path(args.musdr_parent).resolve()))
        subprocess.run([sys.executable, str(ta / 'eval_basic.py'), '--tok2idx_path', str(tok2idx),
                        '--sequences_path', str(seqs), '--output_csv', args.output_csv], check=True, env=env)


if __name__ == '__main__':
    main()
