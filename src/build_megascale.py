#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Build the mega-scale stability dataset (Tsuboyama/Rocklin 2023) from the ThermoMPNN cut.
Single substitutions only, ddG_ML as the label, wild-type sequence as the base, grouped by
WT_name with the predefined protein-clustered train/val/test split preserved."""
import re, numpy as np, pandas as pd

SRC = "/home/user/data_src/ThermoMPNN_Kuhlman-Lab/data_all"
OUT = config.DATA + "/parsed/megascale.csv"
SUB = re.compile(r"^([A-Z])(\d+)([A-Z])$")

def load(path, split):
    head = pd.read_csv(path, nrows=0).columns
    want = [c for c in ["mut_type", "WT_name", "wt_seq", "ddG_ML", "Stabilizing_mut"] if c in head]
    d = pd.read_csv(path, usecols=want)
    has_stab = "Stabilizing_mut" in want
    d = d[d["ddG_ML"].notna() & d["wt_seq"].notna()]
    rows = []
    for r in d.itertuples():
        m = SUB.match(str(r.mut_type))
        if not m:
            continue
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        seq = r.wt_seq
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
            continue
        try:
            val = float(r.ddG_ML)
        except (ValueError, TypeError):
            continue
        stab = bool(r.Stabilizing_mut) if has_stab else (val > 0)
        rows.append((split, r.WT_name, seq, pos - 1, wt, mut, val, stab))
    return rows

def main():
    rows = []
    rows += load(f"{SRC}/training/mega_train.csv", "train")
    rows += load(f"{SRC}/training/mega_val.csv", "val")
    rows += load(f"{SRC}/testing/mega_test.csv", "test")
    df = pd.DataFrame(rows, columns=["split", "protein", "seq", "pos0", "wt", "mut", "value", "stab"])
    df = df.drop_duplicates(subset=["protein", "pos0", "wt", "mut", "split"])
    df.to_csv(OUT, index=False, columns=["split", "protein", "seq", "pos0", "wt", "mut", "value"])

    n = len(df)
    print(f"mega-scale single substitutions: {n}  ({df.protein.nunique()} WT domains)")
    for s in ["train", "val", "test"]:
        sub = df[df.split == s]
        print(f"  {s:5s}: {len(sub):6d} mutations, {sub.protein.nunique()} domains")
    # sign convention: does Stabilizing_mut == (ddG_ML > 0)?
    pos_frac = (df[df.stab].value > 0).mean()
    print(f"sign check: of Stabilizing_mut rows, {pos_frac:.1%} have value>0 "
          f"({'positive = stabilizing' if pos_frac > 0.5 else 'negative = stabilizing'})")
    # protein-blind split check
    tr = set(df[df.split == 'train'].protein); te = set(df[df.split == 'test'].protein)
    print(f"train/test domain overlap: {len(tr & te)} (0 => protein-blind)")

if __name__ == "__main__":
    main()
