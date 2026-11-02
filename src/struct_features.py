#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Structure-aware, anti-symmetry-preserving features.

From the (experimental) structure we read two swap-invariant environment descriptors at the
mutated position: relative solvent accessibility (burial) and heavy-atom contact number.
Each is multiplied by a property DELTA (which flips sign under wt<->mut), so every emitted feature
still negates under the swap and the exact anti-symmetry is preserved.

Output: struct_inter.npz with an [N, 14] interaction block aligned to the dataset CSV row order.
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, ShrakeRupley

# Tien et al. 2013 theoretical max ASA (A^2) for relative SASA
MAXASA = {'A':129.0,'R':274.0,'N':195.0,'D':193.0,'C':167.0,'E':223.0,'Q':225.0,'G':104.0,
          'H':224.0,'I':197.0,'L':201.0,'K':236.0,'M':224.0,'F':240.0,'P':159.0,'S':155.0,
          'T':172.0,'W':285.0,'Y':263.0,'V':174.0}
from features import KD, VOL, CHG, HELIX, SHEET, FLEX, NDON, NACC

T2O = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
       'TYR':'Y','VAL':'V','MSE':'M','SEC':'C','KCX':'K','CSO':'C','CME':'C','PTR':'Y','SEP':'S','TPO':'T'}

INTER_DELTAS = ["hyd", "vol", "chg", "helix", "sheet", "flex", "hbond"]

def deltas(wt, mut):
    g = lambda s, a: s.get(a, 0.0)
    return {
        "hyd": g(KD, mut) - g(KD, wt), "vol": g(VOL, mut) - g(VOL, wt),
        "chg": g(CHG, mut) - g(CHG, wt), "helix": g(HELIX, mut) - g(HELIX, wt),
        "sheet": g(SHEET, mut) - g(SHEET, wt), "flex": g(FLEX, mut) - g(FLEX, wt),
        "hbond": (g(NDON, mut) + g(NACC, mut)) - (g(NDON, wt) + g(NACC, wt)),
    }

_cache = {}
def chain_env(pdb, chain, pdbdir):
    """Return list of (one_letter, rSASA, contact_number) for observed residues, in order."""
    key = (pdb, chain)
    if key in _cache: return _cache[key]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb, f"{pdbdir}/{pdb.upper()}.pdb")
    model = next(iter(structure))
    try:
        ShrakeRupley().compute(model, level="R")
    except Exception:
        _cache[key] = []; return []
    if chain not in [c.id for c in model]:
        _cache[key] = []; return []
    ch = model[chain]
    res = []
    for r in ch:
        if not r.has_id("CA"): continue
        one = T2O.get(r.resname.strip())
        if one is None: continue
        cb = r["CB"].coord if r.has_id("CB") else r["CA"].coord
        sasa = getattr(r, "sasa", np.nan)
        rsasa = min(1.0, sasa / MAXASA.get(one, 200.0)) if sasa == sasa else np.nan
        res.append((one, rsasa, cb))
    # contact numbers (CB within 10 A, |i-j|>2)
    coords = np.array([c for _, _, c in res], dtype=np.float64)
    out = []
    for i, (one, rsasa, _) in enumerate(res):
        if len(coords):
            d = np.linalg.norm(coords - coords[i], axis=1)
            cn = int(np.sum((d < 10.0) & (np.abs(np.arange(len(res)) - i) > 2)))
        else:
            cn = 0
        out.append((one, rsasa, cn))
    _cache[key] = out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA + "/parsed/ddg_dataset.csv")
    ap.add_argument("--out", default=config.WORK + "/struct_inter.npz")
    ap.add_argument("--pdbdir", default="/home/user/data_src/pdbs")
    a = ap.parse_args()
    df = pd.read_csv(a.data).reset_index(drop=True)
    names = [f"{d}_x_{e}" for d in INTER_DELTAS for e in ["burial", "contacts"]]
    X = np.zeros((len(df), len(names)), dtype=np.float32)
    ok = 0
    for k, r in enumerate(df.itertuples()):
        try:
            env = chain_env(r.pdb, r.chain, a.pdbdir)
        except Exception:
            continue
        i = int(r.pos0)
        if i >= len(env): continue
        one, rsasa, cn = env[i]
        if one != r.wt or rsasa != rsasa:  # wt mismatch or NaN sasa -> leave zeros
            continue
        d = deltas(r.wt, r.mut)
        burial = 1.0 - rsasa
        contacts = cn / 20.0
        vec = []
        for dn in INTER_DELTAS:
            vec += [d[dn] * burial, d[dn] * contacts]
        X[k] = vec; ok += 1
    np.savez(a.out, X=X, names=names)
    print(f"struct features: {ok}/{len(df)} rows mapped -> {a.out}  ({len(names)} features)")

if __name__ == "__main__":
    main()
