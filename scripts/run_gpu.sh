#!/usr/bin/env bash
# Turnkey GPU pipeline: large protein-language-model embeddings + zero-shot + structure features,
# then train & evaluate the structure-aware anti-symmetric model. Auto-uses CUDA if present.
# Prereq: pip install -r requirements.txt && bash scripts/get_data.sh
# Pick the model by VRAM: 650M (~16GB) default; 3B on a big GPU via ESM_MODEL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/src"
MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"   # or facebook/esm2_t36_3B_UR50D
echo "Using model: $MODEL"

python3 esm_features.py --model "$MODEL" --data ../data/parsed/ddg_dataset.csv  --out ../artifacts/esm_delta.npz
python3 esm_features.py --model "$MODEL" --data ../data/parsed/s669_dataset.csv --out ../artifacts/esm_delta_s669.npz
python3 zeroshot.py     --model "$MODEL" --data ../data/parsed/ddg_dataset.csv  --out ../artifacts/llr.npz
python3 zeroshot.py     --model "$MODEL" --data ../data/parsed/s669_dataset.csv --out ../artifacts/llr_s669.npz
python3 struct_features.py --data ../data/parsed/ddg_dataset.csv  --out ../artifacts/struct_inter.npz      --pdbdir ../data/pdbs
python3 struct_features.py --data ../data/parsed/s669_dataset.csv --out ../artifacts/struct_inter_s669.npz --pdbdir ../data/pdbs
# ProteinMPNN inverse-folding feature (uses ../data/ProteinMPNN cloned by get_data.sh)
python3 proteinmpnn_features.py --data ../data/parsed/ddg_dataset.csv  --out ../artifacts/mpnn_llr.npz      --pdbdir ../data/pdbs
python3 proteinmpnn_features.py --data ../data/parsed/s669_dataset.csv --out ../artifacts/mpnn_llr_s669.npz --pdbdir ../data/pdbs
python3 train_struct.py llr.npz
echo "done -> results/results_struct.json"
