# Symbolic music generation on POP909

Code for the three parts of the assignment:

- **Task a**: unconditional generation of 32-bar pieces.
- **Task b**: continuing an 8-bar prompt to 32 bars.
- **Bonus**: generating a PIANO accompaniment for a 16-bar melody.

All models are trained only on the TA's POP909 training split.

| Part | Model |
|---|---|
| Task a / b | REMI tokens with all tracks merged; 8-layer Llama decoder (d=512, 33.9M parameters) |
| Bonus | Interleaved melody/accompaniment bars with one bar of melody lookahead and a chord token per bar (`bonus_chord_k1`). Chords come from a separate melody→chord predictor (an ensemble of 4 bidirectional Transformer encoders). While sampling the piano, pitches outside the bar's chord get logit −4. |

## Setup

```bash
conda create -n mgen python=3.10 -y && conda activate mgen
pip install -r requirements.txt
# for .wav output only:
conda install -c conda-forge fluidsynth -y    # or: apt install fluidsynth
```

`midi2wav.sh` looks for the SoundFont at `~/soundfonts/GeneralUser-GS.sf2`. To use another one, set `SF2=/path/to/font.sf2`.

## Checkpoints and results

Everything is in one Google Drive folder: https://drive.google.com/drive/folders/1YHXrWcPuy1upCD1di8bChhXhI09sP7ii

- `checkpoints/` holds the models. Download it next to this README, or anywhere else and pass its path with `--ckpt_dir`:
  ```
  checkpoints/
  ├── basic/            # Task a / b model (HF Llama + tokenizer_remi.json)
  ├── bonus_chord_k1/   # bonus accompaniment model
  └── chord_pred/       # 4 chord predictors, used together as an ensemble
      ├── d192_l4_dr0.1/  d256_l6_dr0/  d384_l8_dr0/  d384_l8_dr0.1/
  ```
- `task_a/`, `task_b/` and `bonus/` hold the submitted `.mid` and `.wav` files.

## Inference (one command)

```bash
bash run_inference.sh --ckpt_dir checkpoints --prompt_root POP909/prompts --output_dir out --wav
```

| Option | Meaning |
|---|---|
| `--ckpt_dir` | the downloaded `checkpoints/` folder |
| `--prompt_root` | the TA prompt folder that contains `basic-b/` and `bonus/` |
| `--output_dir` | where results are written: `<output_dir>/task_a`, `task_b`, `bonus` |
| `--task all\|a\|b\|bonus` | run only one part (default `all`) |
| `--wav` | also render every `.mid` to `.wav` |
| `PYTHON=...` | environment variable choosing the interpreter (default `python`) |

A GPU is used when available. The whole run takes about 2 minutes on one H100.

## Inference (individual steps)

Sampling presets are defined in `configs/sampling.json`:

| Preset | temperature | top-k | top-p |
|---|---|---|---|
| `A_t1.0_p0.95` | 1.0 | – | 0.95 |
| `B_t1.2_p0.9` | 1.2 | – | 0.9 |
| `C_t0.8_k10` | 0.8 | 10 | – |

```bash
# Task a: 20 pieces of 32 bars (submitted preset: A)
python generate_basic.py --ckpt checkpoints/basic --config A_t1.0_p0.95 --n 20 --bars 32 --output_dir out/task_a

# Task b: each prompt x each preset (or pass --config to choose presets)
python continue_prompt.py --ckpt checkpoints/basic --prompt_dir POP909/prompts/basic-b --output_dir out/task_b

# Bonus step 1: one chord per bar from the melody (several --ckpt = ensemble)
python chord_predictor.py predict --ckpt checkpoints/chord_pred/* --prompt_dir POP909/prompts/bonus \
    --out out/bonus/predicted_chords.json
# Bonus step 2: accompaniment that follows those chords
python generate_bonus.py --ckpt checkpoints/bonus_chord_k1 --prompt_dir POP909/prompts/bonus \
    --config C_t0.8_k10 --chord_file out/bonus/predicted_chords.json --chord_bias 4 --seed 0 --output_dir out/bonus

# MIDI -> WAV
bash midi2wav.sh out/bonus out/bonus
```

A bonus prompt folder needs `melody.mid` and `beat_midi.txt`. `chord_midi.txt` is optional and is used only to print chord accuracy. The output `<id>.mid` contains the original MELODY track and the generated PIANO track, placed on the prompt's timeline in seconds.

## Evaluation

The metrics come from the TA scripts, which are not included in this repo:

```bash
python eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir out/bonus --output_csv out/bonus/metrics.csv
```

`eval/eval_basic_midis.py` runs `eval_basic.py` (H1 / H4 / GS) over a folder of generated MIDI files.

## Training

Run these from this folder, with the TA data at `../POP909` and generated data written to `../data`. `--batch_size 0` picks the largest batch size that fits on the GPU.

```bash
# Align POP909 to beat_midi.txt (bar lines / time signatures)
python data_process/fix_timesig.py --src ../POP909 --dst ../data/POP909_fixed

# Task a / b
python data_process/tokenize_basic.py --src ../data/POP909_fixed/training \
    --out_dir ../data/tokens_basic --tokenizer_out configs/tokenizer_remi.json
python train.py --data_dir ../data/tokens_basic --tokenizer configs/tokenizer_remi.json --out_dir ../runs/basic \
    --batch_size 0 --train_tokens 4e8 --lr 2e-4 --warmup 300 --eval_every 100 --patience 6

# Bonus accompaniment model (lookahead 1, with chord tokens)
python data_process/tokenize_bonus.py --src ../data/POP909_fixed/training --split ../data/tokens_basic/split.json \
    --lookahead 1 --chords --raw_src ../POP909/training --out_dir ../data/tokens_bonus_chord_k1 --tokenizer_out configs/tokenizer_remi_bonus_chord.json
python train.py --data_dir ../data/tokens_bonus_chord_k1 --tokenizer configs/tokenizer_remi_bonus_chord.json \
    --out_dir ../runs/bonus_chord_k1 --ctx 2048 --bos_windows \
    --batch_size 0 --train_tokens 3e8 --lr 2e-4 --warmup 300 --eval_every 100 --patience 6

# Chord predictors (the 4 ensemble members)
python chord_predictor.py prep --out ../data/chord_pred/data.pkl
python chord_predictor.py train --data ../data/chord_pred/data.pkl --out_dir ../runs/chord_pred/d192_l4_dr0.1 --d 192 --layers 4 --dropout 0.1
python chord_predictor.py train --data ../data/chord_pred/data.pkl --out_dir ../runs/chord_pred/d256_l6_dr0   --d 256 --layers 6 --dropout 0
python chord_predictor.py train --data ../data/chord_pred/data.pkl --out_dir ../runs/chord_pred/d384_l8_dr0   --d 384 --layers 8 --dropout 0
python chord_predictor.py train --data ../data/chord_pred/data.pkl --out_dir ../runs/chord_pred/d384_l8_dr0.1 --d 384 --layers 8 --dropout 0.1

# Bonus-style prompts from the validation songs, used to choose --chord_bias
python data_process/make_val_prompts.py --raw_src ../POP909/training --split ../data/tokens_basic/split.json \
    --out_dir ../data/val_prompts_bonus
```

Each script's docstring has a usage example, and `--help` lists its arguments.
