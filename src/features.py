#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Anti-symmetric feature construction for ddG-of-folding prediction.

Every feature is built so that swapping (wild-type <-> mutant) negates it:
  - property DELTAS            d_p = scale[mut] - scale[wt]        (one delta -> flips sign)
  - DELTA x ENVIRONMENT        d_p * env_q                         (env is swap-invariant -> flips sign)
No raw environment term and no delta*delta term appears, so the full vector F obeys
F(mut->wt) = -F(wt->mut) exactly.  Any learner h then yields an exactly anti-symmetric
predictor via  pred = (h(F) - h(-F)) / 2 , matching the thermodynamic law
ddG(wt->mut) = -ddG(mut->wt).
"""
import numpy as np

AAs = "ACDEFGHIKLMNPQRSTVWY"

# --- residue property scales (published values) ---
KD = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
      'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
VOL = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,
       'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,
       'Y':193.6,'V':140.0}
CHG = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}
POL = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'Q':10.5,'E':12.3,'G':9.0,'H':10.4,'I':5.2,
       'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9}
# Pace & Scholtz helix propensity (kcal/mol; lower => stronger helix former). Negate so higher=favoring.
HELIX = {'A':0.0,'L':0.21,'R':0.21,'M':0.24,'K':0.26,'Q':0.39,'E':0.40,'I':0.41,'W':0.49,'S':0.50,
         'Y':0.53,'F':0.54,'H':0.61,'V':0.61,'N':0.65,'T':0.66,'C':0.68,'D':0.69,'G':1.00,'P':3.16}
HELIX = {k: -v for k, v in HELIX.items()}
# Chou-Fasman sheet propensity (higher => stronger sheet former)
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,'I':1.60,
         'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
# Vihinen average residue flexibility (higher => more flexible)
FLEX = {'A':0.360,'R':0.530,'N':0.460,'D':0.510,'C':0.350,'Q':0.490,'E':0.500,'G':0.540,'H':0.320,
        'I':0.460,'L':0.370,'K':0.470,'M':0.300,'F':0.310,'P':0.510,'S':0.510,'T':0.440,'W':0.310,
        'Y':0.420,'V':0.390}
NDON = {'R':3,'K':1,'W':1,'N':1,'Q':1,'H':1,'S':1,'T':1,'Y':1,'C':1}   # side-chain H-bond donors
NACC = {'D':2,'E':2,'N':1,'Q':1,'H':1,'S':1,'T':1,'Y':1}               # side-chain H-bond acceptors
AROM = {'F':1,'W':1,'Y':1,'H':1}
MW = {'G':57.05,'A':71.08,'S':87.08,'P':97.12,'V':99.13,'T':101.10,'C':103.14,'L':113.16,'I':113.16,
      'N':114.10,'D':115.09,'Q':128.13,'K':128.17,'E':129.12,'M':131.19,'H':137.14,'F':147.18,
      'R':156.19,'Y':163.18,'W':186.21}

def g(scale, a, default=0.0):
    return scale.get(a, default)

# property scales used for per-residue DELTA features
_DELTA_SCALES = [("hyd", KD), ("vol", VOL), ("chg", CHG), ("pol", POL),
                 ("helix", HELIX), ("sheet", SHEET), ("flex", FLEX),
                 ("ndon", NDON), ("nacc", NACC), ("arom", AROM), ("mw", MW)]

# deltas used in interactions, and the environment descriptors they multiply
_INTER_DELTAS = ["hyd", "vol", "chg", "helix", "sheet", "flex", "hbond"]
_INTER_ENV = ["burial", "envchg", "term"]

WIN = 5  # +/- window for environment

def _env(seq, i):
    """Swap-invariant environment descriptors from neighbors (excluding position i)."""
    lo, hi = max(0, i - WIN), min(len(seq), i + WIN + 1)
    neigh = [seq[j] for j in range(lo, hi) if j != i and seq[j] in KD]
    n = max(len(neigh), 1)
    burial = sum(g(KD, a) for a in neigh) / n                 # hydrophobic env ~ buried
    envchg = sum(g(CHG, a) for a in neigh) / n                # local charge density
    L = len(seq)
    term = min(i, L - 1 - i) / (L / 2.0 + 1e-9)               # 0 at termini -> 1 at center
    return {"burial": burial, "envchg": envchg, "term": term}

FEATURE_NAMES = (
    [f"d_{name}" for name, _ in _DELTA_SCALES]
    + [f"d_{d}_x_{e}" for d in _INTER_DELTAS for e in _INTER_ENV]
)

def _deltas(wt, mut):
    d = {}
    for name, scale in _DELTA_SCALES:
        d[name] = g(scale, mut) - g(scale, wt)
    d["hbond"] = (g(NDON, mut) + g(NACC, mut)) - (g(NDON, wt) + g(NACC, wt))
    return d

def featurize(seq, i, wt, mut):
    d = _deltas(wt, mut)
    env = _env(seq, i)
    feats = [d[name] for name, _ in _DELTA_SCALES]
    for dn in _INTER_DELTAS:
        for en in _INTER_ENV:
            feats.append(d[dn] * env[en])
    return np.asarray(feats, dtype=np.float64)

def featurize_df(df):
    X = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float64)
    for k, (_, row) in enumerate(df.iterrows()):
        X[k] = featurize(row["seq"], int(row["pos0"]), row["wt"], row["mut"])
    return X

if __name__ == "__main__":
    # sanity: F(mut->wt) == -F(wt->mut)
    s = "MKVLYACDEFGHIKLMNPQRSTVWY" * 4
    f1 = featurize(s, 10, s[10], "P")
    f2 = featurize(s[:10] + "P" + s[11:], 10, "P", s[10])
    err = np.max(np.abs(f1 + f2))
    print("n_features =", len(FEATURE_NAMES))
    print("max |F + F_rev| =", err, "(exact anti-symmetry)" if err < 1e-9 else "BROKEN")
