#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Build a second, INDEPENDENT mutational-stability dataset from FireProtDB (sequences inline,
no structure reconstruction needed). Used to show the anti-symmetric recipe generalizes across
datasets, and that the trainer is property-agnostic (swap the target column for any property)."""
import glob, pandas as pd

FP = "/home/user/data_src/ThermoMPNN_Kuhlman-Lab/data_all"
OUT = config.DATA + "/parsed/fireprot_ddg.csv"

def main():
    frames = []
    for f in [FP + "/training/fireprot_train.csv", FP + "/training/fireprot_val.csv",
              FP + "/testing/fireprot_test.csv"]:
        frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    d = d[d["ddG"].notna() & d["sequence"].notna()]
    rows = []
    for r in d.itertuples():
        seq = r.sequence; pos = int(r.position)
        if pos < 1 or pos > len(seq) or seq[pos - 1] != r.wild_type:
            continue
        rows.append(dict(dataset="FireProt", protein=r.uniprot_id, pos0=pos - 1,
                         wt=r.wild_type, mut=r.mutation, value=float(r.ddG), seq=seq))
    out = pd.DataFrame(rows).drop_duplicates(subset=["protein", "pos0", "wt", "mut"])
    out.to_csv(OUT, index=False)
    print(f"FireProt ddG: {len(out)} verified mutations, {out.protein.nunique()} proteins -> {OUT}")

    # independence from S2648 training proteins (by sequence)
    s2 = pd.read_csv(config.DATA + "/parsed/ddg_dataset.csv")
    s2seq = set(s2[s2.dataset == "S2648"].seq)
    exact = sum(1 for s in set(out.seq) if s in s2seq)
    print(f"FireProt sequences exactly present in S2648: {exact}/{out.seq.nunique()}")

if __name__ == "__main__":
    main()
