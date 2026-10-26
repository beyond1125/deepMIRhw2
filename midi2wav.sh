#!/bin/bash
# 用法：bash midi2wav.sh <midi資料夾> <wav輸出資料夾>
IN_DIR="$1"
OUT_DIR="$2"
SF2="${SF2:-$HOME/soundfonts/GeneralUser-GS.sf2}"
# 優先用環境變數 FLUIDSYNTH，其次 PATH
FS="${FLUIDSYNTH:-$(command -v fluidsynth)}"
if [ ! -x "$FS" ]; then
  echo "找不到 fluidsynth：$FS"
  exit 1
fi

if [ ! -d "$IN_DIR" ]; then
  echo "找不到輸入資料夾：$IN_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"
count=0
for f in "$IN_DIR"/*.mid "$IN_DIR"/*.midi; do
  [ -e "$f" ] || continue
  name=$(basename "$f"); name="${name%.*}"
  if "$FS" -ni -q -F "$OUT_DIR/$name.wav" -r 44100 "$SF2" "$f" > /dev/null 2>&1; then
    echo "OK   $name"
    count=$((count+1))
  else
    echo "FAIL $name"
  fi
done
echo "完成，共轉換 $count 個檔案"