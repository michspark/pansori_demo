#!/bin/bash
# Run the Mel+PESTO+MIDI ensemble (weights 0.5/0.1/0.4) on every version-split
# test song, writing <hash>_ensemble.csv into $OUT. Run from anywhere.
#   MODEL_REPO=../PansoriModeClassification DATA_ROOT=../Pansori_Data OUT=work bash tools/run_test_inference.sh
set -euo pipefail
MODEL_REPO=$(realpath "${MODEL_REPO:?}")
DATA_ROOT=$(realpath "${DATA_ROOT:?}")
OUT=$(realpath -m "${OUT:?}")
PY="$MODEL_REPO/.venv/bin/python"
mkdir -p "$OUT"
cd "$MODEL_REPO"
MEL_DIR=$(dirname weights/version_retrain/mel/*/fold1_best_model.pt)
PESTO_DIR=$(dirname weights/version_retrain/pesto/*/fold1_best_model.pt)
MIDI_DIR=$(dirname weights/version_retrain/midi/*/fold1_best_model.pt)
while read -r stem; do
  [ -z "$stem" ] && continue
  hash=${stem%%-*}
  [ -f "$OUT/${hash}_ensemble.csv" ] && continue
  "$PY" infer/infer_ensemble.py --quiet \
    --mel   "$DATA_ROOT/Audio_Original/$stem.wav"            --mel_dir   "$MEL_DIR" \
    --pesto "$DATA_ROOT/Audio_pesto_output/${stem}_vocal.f0.csv" --pesto_dir "$PESTO_DIR" \
    --midi  "$DATA_ROOT/rosvot_midi/${stem}_vocal.mid"       --midi_dir  "$MIDI_DIR" \
    --weights 0.5 0.1 0.4 --name "$hash" --out "$OUT"
done < data/splits/pansori_version_split/test.txt
