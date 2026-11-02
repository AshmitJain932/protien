#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Build a verified sequence-level ddG dataset from ACDC-NN .mut files + RCSB PDBs.

Output: parsed CSV with columns
    dataset, protein (pdbchain), pdb, chain, pos0 (0-based), wt, mut, ddg, seq, fold
Position convention (validated on 1A43): 1-based sequential over observed CA residues.
"""
import os, glob, csv, sys
from collections import defaultdict

SRC = config.DATA + ""
PDB = f"{SRC}/pdbs"
OUT = f"{SRC}/parsed"
os.makedirs(OUT, exist_ok=True)

THREE2ONE = {
 'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
 'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
 'THR':'T','TRP':'W','TYR':'Y','VAL':'V',
 # common modified residues observed in these structures
 'MSE':'M','SEC':'U','PYL':'O','HYP':'P','PCA':'E','SEP':'S','TPO':'T','PTR':'Y',
 'CSO':'C','CME':'C','CSD':'C','KCX':'K','LLP':'K','MLY':'K','M3L':'K','FME':'M',
 'CAS':'C','OCS':'C','CSS':'C','CSX':'C','NEP':'H','HIC':'H','MHO':'M',
}

_seq_cache = {}

def chain_sequence(pdb_id, chain):
    """Return (seq, resseq_list) for observed residues of a chain, first model only."""
    key = (pdb_id, chain)
    if key in _seq_cache:
        return _seq_cache[key]
    path = f"{PDB}/{pdb_id.upper()}.pdb"
    seq = []
    seen = set()
    in_model = True  # before any MODEL line we are in the (only) model
    started_model = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                if started_model:
                    in_model = False   # second model onward -> ignore
                else:
                    started_model = True
                    in_model = True
                continue
            if rec == "ENDMDL":
                continue
            if rec in ("ATOM", "HETATM") and in_model:
                if line[12:16].strip() != "CA":
                    continue
                if line[21] != chain:
                    continue
                alt = line[16]
                if alt not in (" ", "A"):
                    continue
                resname = line[17:20].strip()
                one = THREE2ONE.get(resname)
                if one is None or one in ("U", "O"):
                    one = "X"
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26].strip()
                k = (resseq, icode)
                if k in seen:
                    continue
                seen.add(k)
                seq.append(one)
    s = "".join(seq)
    _seq_cache[key] = s
    return s

def load_mut_file(path, dataset, fold):
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        tag, mut, ddg = parts[0], parts[1], parts[2]
        pdb_id, chain = tag[:4], tag[4]
        wt, mt = mut[0], mut[-1]
        try:
            pos = int(mut[1:-1]); ddg = float(ddg)
        except ValueError:
            continue
        rows.append((dataset, f"{pdb_id}{chain}", pdb_id, chain, pos, wt, mt, ddg, fold))
    return rows

def main():
    all_rows = []
    # S2648 with predefined 10-fold test splits (fold = k from TS_k)
    for k in range(10):
        for r in load_mut_file(f"{SRC}/acdc-nn/datasets/s2648/s2648_TS_{k}.mut", "S2648", k):
            all_rows.append(r)
    # Ssym direct (fold = -1), inverse (fold = -2)
    for f in sorted(glob.glob(f"{SRC}/acdc-nn/datasets/Ssym/ssym_TS_dir_*.mut")):
        all_rows += load_mut_file(f, "Ssym_dir", -1)
    for f in sorted(glob.glob(f"{SRC}/acdc-nn/datasets/Ssym/ssym_TS_inv_*.mut")):
        all_rows += load_mut_file(f, "Ssym_inv", -2)

    # de-dup identical (dataset,protein,pos,wt,mut)
    seen = set(); dedup = []
    for r in all_rows:
        key = (r[0], r[1], r[4], r[5], r[6])
        if key in seen:
            continue
        seen.add(key); dedup.append(r)

    out = []
    stats = defaultdict(lambda: [0, 0])  # dataset -> [matched, total]
    missing_pdb = set()
    for (dataset, protein, pdb_id, chain, pos, wt, mt, ddg, fold) in dedup:
        seq = chain_sequence(pdb_id, chain)
        stats[dataset][1] += 1
        if not seq:
            missing_pdb.add(pdb_id); continue
        if pos < 1 or pos > len(seq):
            continue
        if seq[pos-1] != wt:
            continue
        stats[dataset][0] += 1
        out.append(dict(dataset=dataset, protein=protein, pdb=pdb_id, chain=chain,
                        pos0=pos-1, wt=wt, mut=mt, ddg=ddg, seq=seq, fold=fold))

    with open(f"{OUT}/ddg_dataset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset","protein","pdb","chain","pos0","wt","mut","ddg","seq","fold"])
        w.writeheader()
        for r in out:
            w.writerow(r)

    print("=== match rate by dataset (wt residue verified against structure) ===")
    for ds, (m, t) in sorted(stats.items()):
        print(f"  {ds:10s}  {m}/{t}  = {100*m/max(t,1):.1f}%")
    print("total written:", len(out))
    uprot = len(set(r['protein'] for r in out if r['dataset']=='S2648'))
    print("S2648 unique proteins:", uprot)
    if missing_pdb:
        print("missing/empty PDBs:", len(missing_pdb), sorted(missing_pdb)[:10])

if __name__ == "__main__":
    main()
