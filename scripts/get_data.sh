#!/usr/bin/env bash
# Reproduce the parsed datasets from scratch: clone benchmark sources, download PDB
# structures, then build + verify S2648 / Ssym / S669. Requires network access.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA/pdbs"

# 1. benchmark sources
[ -d "$DATA/acdc-nn" ] || git clone --depth 1 https://github.com/compbiomed-unito/acdc-nn "$DATA/acdc-nn"
[ -d "$DATA/ThermoMPNN_Kuhlman-Lab" ] || git clone --depth 1 https://github.com/Kuhlman-Lab/ThermoMPNN "$DATA/ThermoMPNN_Kuhlman-Lab"

# 2. unique PDB ids across all three sets
python3 - "$DATA" <<'PY'
import sys, glob, pandas as pd
DATA = sys.argv[1]; ids = set()
for f in glob.glob(f"{DATA}/acdc-nn/datasets/s2648/*.mut") + glob.glob(f"{DATA}/acdc-nn/datasets/Ssym/*.mut"):
    for line in open(f):
        p = line.split()
        if p: ids.add(p[0][:4].upper())
s669 = pd.read_csv(f"{DATA}/ThermoMPNN_Kuhlman-Lab/data_all/testing/s669_clean_dir.csv")
ids |= {p[:4].upper() for p in s669.PDB}
open(f"{DATA}/pdb_ids.txt", "w").write("\n".join(sorted(ids)))
print(len(ids), "unique PDB ids")
PY

# 3. download structures (parallel)
cat "$DATA/pdb_ids.txt" | xargs -P 8 -I{} bash -c '
  f="'"$DATA"'/pdbs/{}.pdb"
  [ -s "$f" ] || curl -s --max-time 60 "https://files.rcsb.org/download/{}.pdb" -o "$f"'
echo "downloaded $(ls "$DATA"/pdbs/*.pdb | wc -l) structures"

# 4. build + verify parsed datasets
cd "$ROOT/src"
python3 build_dataset.py
python3 build_s669.py
echo "done -> data/parsed/"
