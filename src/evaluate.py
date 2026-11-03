#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Rigorous, honest evaluation on S669: full set vs the homology-controlled clean-80 subset
(src/splits.py), with multi-seed variance AND bootstrap 95% CIs. Covers the feature sets that
are reproducible on CPU from the committed data + saved 35M/150M feature caches
(biophysical, and bio+ESM+LLR). The 650M/structure/ProteinMPNN numbers require a GPU run and are
not re-evaluated here (their feature artifacts are not committed)."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

D = config.DATA + "/parsed"
SEEDS = [0, 1, 2, 3, 4]
B = 1000  # bootstrap resamples

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def xgb(seed): return XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=1.5, min_child_weight=3, n_jobs=4, random_state=seed)

def per_protein_spearman(y, p, prot):
    df = pd.DataFrame({"y": y, "p": p, "prot": prot})
    rs = [spearmanr(g.y, g.p)[0] for _, g in df.groupby("prot") if len(g) >= 5]
    rs = [x for x in rs if x == x]
    return float(np.mean(rs)) if rs else float("nan")

def boot_ci(y, p, fn=lambda a, b: pearsonr(a, b)[0], B=B, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try: vals.append(fn(y[idx], p[idx]))
        except Exception: pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def main():
    ddg = pd.read_csv(f"{D}/ddg_dataset.csv"); m2 = (ddg.dataset == "S2648").values
    s = ddg[m2].reset_index(drop=True); y = s.ddg.values
    bio = F.featurize_df(s).astype(np.float32)
    emb = np.load(config.WORK + "/esm_delta.npz")["delta"][m2]
    llr = np.load(config.WORK + "/llr.npz")["llr"][m2][:, None]

    te = pd.read_csv(f"{D}/s669_dataset.csv").reset_index(drop=True); yte = te.ddg.values
    bio_te = F.featurize_df(te).astype(np.float32)
    emb_te = np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    llr_te = np.load(config.WORK + "/llr_s669.npz")["llr"][:, None]
    clean_ids = set(open(f"{D}/s669_clean.txt").read().split())
    clean = te.protein.isin(clean_ids).values
    print(f"S669 full N={len(te)} ({te.protein.nunique()} proteins); "
          f"clean N={clean.sum()} ({te[clean].protein.nunique()} proteins)")

    sets = {"bio": (bio, bio_te), "bio+esm+llr": (np.hstack([bio, emb, llr]), np.hstack([bio_te, emb_te, llr_te]))}
    out = {}
    for name, (X, Xte) in sets.items():
        preds = np.stack([anti(xgb(sd).fit(*augment(X, y)), Xte) for sd in SEEDS], 0)  # [seeds, N]
        pmean = preds.mean(0)
        rec = {}
        for tag, mask in [("full", np.ones(len(te), bool)), ("clean80", clean)]:
            yy, pp = yte[mask], pmean[mask]
            seed_r = [pearsonr(yte[mask], preds[i][mask])[0] for i in range(len(SEEDS))]
            lo, hi = boot_ci(yy, pp)
            rec[tag] = dict(
                pearson_mean=float(np.mean(seed_r)), pearson_sd=float(np.std(seed_r)),
                pearson_boot95=[round(lo, 3), round(hi, 3)],
                spearman=float(spearmanr(yy, pp)[0]), rmse=rmse(yy, pp),
                per_protein_spearman=per_protein_spearman(yy, pp, te.protein.values[mask]),
                n=int(mask.sum()))
        out[name] = rec
        print(f"\n### {name}")
        for tag in ["full", "clean80"]:
            r = rec[tag]
            print(f"  S669 {tag:8s}: r={r['pearson_mean']:.3f}±{r['pearson_sd']:.3f} "
                  f"boot95={r['pearson_boot95']} rho={r['spearman']:.3f} rmse={r['rmse']:.2f} "
                  f"per-prot rho={r['per_protein_spearman']:.3f} (N={r['n']})")
    json.dump(out, open(config.RESULTS + "/results_clean_eval.json", "w"), indent=2)
    print("\nsaved results_clean_eval.json")

if __name__ == "__main__":
    main()
