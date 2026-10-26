#!/bin/bash
# One-command inference for all three tasks with the submitted settings.
#
# Usage:
#   bash run_inference.sh --ckpt_dir checkpoints --prompt_root POP909/prompts --output_dir out [--task all|a|b|bonus] [--wav]
#
#   --ckpt_dir     folder downloaded from the Drive link (basic/, bonus_chord_k1/, chord_pred/)
#   --prompt_root  TA prompt folder containing basic-b/ and bonus/
#   --output_dir   where results are written: <output_dir>/task_a, task_b, bonus
#   --task         which task to run (default: all)
#   --wav          also render every .mid to .wav with midi2wav.sh (needs fluidsynth + a SoundFont)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

CKPT_DIR=checkpoints
PROMPT_ROOT=POP909/prompts
OUT_DIR=out
TASK=all
WAV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt_dir) CKPT_DIR="$2"; shift 2 ;;
        --prompt_root) PROMPT_ROOT="$2"; shift 2 ;;
        --output_dir) OUT_DIR="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --wav) WAV=1; shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

render() {
    [ $WAV = 1 ] && bash midi2wav.sh "$1" "$1"
    return 0
}

if [ $TASK = all ] || [ $TASK = a ]; then
    # Task a: 20 unconditional 32-bar pieces with the submitted preset
    $PY generate_basic.py --ckpt "$CKPT_DIR/basic" --config A_t1.0_p0.95 --n 20 --bars 32 --seed 0 \
        --output_dir "$OUT_DIR/task_a"
    render "$OUT_DIR/task_a"
fi

if [ $TASK = all ] || [ $TASK = b ]; then
    # Task b: every prompt continued to 32 bars with each of the 3 presets
    $PY continue_prompt.py --ckpt "$CKPT_DIR/basic" --prompt_dir "$PROMPT_ROOT/basic-b" --seed 0 \
        --output_dir "$OUT_DIR/task_b"
    render "$OUT_DIR/task_b"
fi

if [ $TASK = all ] || [ $TASK = bonus ]; then
    # Bonus step 1: chords per bar from the melody, ensemble of the 4 chord predictors
    mkdir -p "$OUT_DIR/bonus"
    $PY chord_predictor.py predict --ckpt "$CKPT_DIR"/chord_pred/* --prompt_dir "$PROMPT_ROOT/bonus" \
        --out "$OUT_DIR/bonus/predicted_chords.json"
    # Bonus step 2: piano accompaniment that follows those chords (non-chord pitches get logit - 4)
    $PY generate_bonus.py --ckpt "$CKPT_DIR/bonus_chord_k1" --prompt_dir "$PROMPT_ROOT/bonus" \
        --config C_t0.8_k10 --chord_file "$OUT_DIR/bonus/predicted_chords.json" --chord_bias 4 --seed 0 \
        --output_dir "$OUT_DIR/bonus"
    render "$OUT_DIR/bonus"
fi
