#!/usr/bin/env bash
# Mega-scale training (Tsuboyama/Rocklin ~270k single mutations) with large-PLM features on a GPU.
# Run AFTER scripts/run_gpu.sh (which produces the 650M S669 features used for the transfer test).
# The per-variant embedding step is the heavy part (~270k forwards) — budget ~40-60 min on a T4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/src"
MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
echo "Using model: $MODEL"

python3 build_megascale.py
python3 zeroshot.py     --model "$MODEL" --data ../data/parsed/megascale.csv --out ../artifacts/llr_mega.npz
python3 esm_features.py  --model "$MODEL" --data ../data/parsed/megascale.csv --out ../artifacts/esm_delta_mega.npz
python3 train_megascale.py   # auto-uses the mega embeddings if present; saves models/model_mega.json
if [ -n "${PERSIST_DIR:-}" ]; then
  mkdir -p "$PERSIST_DIR"
  cp -r ../models "$PERSIST_DIR"/ 2>/dev/null || true
  cp -r ../artifacts "$PERSIST_DIR"/ 2>/dev/null || true
  echo "persisted models + artifacts to $PERSIST_DIR"
fi
echo "done -> results/results_megascale.json (+ models/model_mega.json, model_cards_mega.json)"
