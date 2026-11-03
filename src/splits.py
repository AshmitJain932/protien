#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Sequence-identity clustering of all proteins (S2648 training + S669 test) and a FROZEN,
homology-controlled clean test set.

Why: the repo previously called S669 "zero overlap" using only exact-sequence and 6-mer Jaccard
checks, which cannot detect evolutionary homology. Here we compute real local-alignment identity
between every pair of proteins, single-linkage cluster at a 30% identity threshold, and define
the clean test as the S669 proteins whose cluster contains NO training protein. Whole clusters
never straddle the train/test boundary.

Outputs (committed, small, deterministic):
  data/parsed/protein_clusters.tsv  protein, dataset, cluster_id, max_identity_to_S2648
  data/parsed/s669_clean.txt        S669 protein ids with <30% identity to any S2648 protein
"""
import numpy as np, pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

D = config.DATA + "/parsed"
THRESH = 30.0  # % identity; clusters linked at or above this

A = PairwiseAligner(); A.mode = "local"
A.substitution_matrix = substitution_matrices.load("BLOSUM62")
A.open_gap_score = -11; A.extend_gap_score = -1

def identity(a, b):
    aln = A.align(a, b)[0]; m = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        m += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return 100.0 * m / min(len(a), len(b))

def main():
    ddg = pd.read_csv(f"{D}/ddg_dataset.csv")
    s2 = ddg[ddg.dataset == "S2648"][["protein", "seq"]].drop_duplicates("protein")
    s6 = pd.read_csv(f"{D}/s669_dataset.csv")[["protein", "seq"]].drop_duplicates("protein")
    prot = ([(p, s, "S2648") for p, s in zip(s2.protein, s2.seq)]
            + [(p, s, "S669") for p, s in zip(s6.protein, s6.seq)])
    n = len(prot)
    seqs = [s for _, s, _ in prot]
    print(f"clustering {n} proteins ({(s2.shape[0])} S2648 + {s6.shape[0]} S669) at {THRESH}% identity ...", flush=True)

    # union-find over all pairs with identity >= THRESH
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if identity(seqs[i], seqs[j]) >= THRESH:
                union(i, j)
    cl = [find(i) for i in range(n)]
    # remap cluster ids to 0..k
    remap = {c: k for k, c in enumerate(sorted(set(cl)))}
    cl = [remap[c] for c in cl]

    # which clusters contain a training protein
    train_clusters = {cl[i] for i in range(n) if prot[i][2] == "S2648"}

    # max identity of each S669 protein to any S2648 protein (for the report column)
    s2_idx = [i for i in range(n) if prot[i][2] == "S2648"]
    rows, clean = [], []
    for i in range(n):
        p, s, ds = prot[i]
        maxid = 0.0
        if ds == "S669":
            maxid = max((identity(s, seqs[j]) for j in s2_idx), default=0.0)
            if cl[i] not in train_clusters:
                clean.append(p)
        rows.append((p, ds, cl[i], round(maxid, 1)))

    pd.DataFrame(rows, columns=["protein", "dataset", "cluster_id", "max_identity_to_S2648"]).to_csv(
        f"{D}/protein_clusters.tsv", sep="\t", index=False)
    open(f"{D}/s669_clean.txt", "w").write("\n".join(clean) + "\n")

    n_s669 = s6.shape[0]
    print(f"total clusters: {len(set(cl))}")
    print(f"S669 proteins: {n_s669} total")
    print(f"  clean (no training protein in cluster, <{THRESH}% identity): {len(clean)}")
    print(f"  homology-contaminated (>= {THRESH}% to training): {n_s669 - len(clean)}")
    print(f"wrote protein_clusters.tsv and s669_clean.txt")

if __name__ == "__main__":
    main()
