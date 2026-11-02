#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Ensemble + calibrated uncertainty for the anti-symmetric ddG model.

- Ensemble of anti-symmetric XGBoost models (seed/subsample diversity) -> point prediction + a
  spread that proxies confidence.
- Split-conformal prediction gives intervals with a *verifiable* coverage guarantee:
  calibrate the nonconformity quantile on a held-out protein-disjoint fold, then test empirical
  coverage both in-distribution (another S2648 fold) and under shift (S669 blind).
- Normalized (adaptive) conformal uses the ensemble spread so intervals widen where the model is
  unsure.

Usage: python uncertainty.py [--llr llr.npz]   (swap in llr650.npz once ready)
"""
import argparse, json, numpy as np, pandas as pd
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import features as F

DDG = config.DATA + "/parsed/ddg_dataset.csv"
S669 = config.DATA + "/parsed/s669_dataset.csv"

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def make_model(seed):
    return XGBRegressor(n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.75,
                        colsample_bytree=0.55, reg_lambda=2.0, min_child_weight=4, n_jobs=4,
                        objective="reg:squarederror", random_state=seed)

def fit_ensemble(X, y, K=8):
    Xa, ya = augment(X, y)
    models = []
    for k in range(K):
        m = make_model(1000 + 7 * k); m.fit(Xa, ya); models.append(m)
    return models

def predict_ensemble(models, X):
    P = np.stack([anti(m, X) for m in models], axis=0)   # [K, N]
    return P.mean(0), P.std(0)                            # mean, spread

def coverage(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--llr", default="llr.npz")
    ap.add_argument("--tag", default="150M"); a = ap.parse_args()

    df = pd.read_csv(DDG); m2 = (df.dataset == "S2648").values
    emb = np.load(config.WORK + "/esm_delta.npz")["delta"]
    llr = np.load(config.WORK + "/" + a.llr)["llr"][:, None]
    Xall = np.hstack([F.featurize_df(df).astype(np.float32), emb, llr])
    s = df[m2].reset_index(drop=True); fs = s.fold.values
    X, y = Xall[m2], s.ddg.values

    tr = np.isin(fs, range(8)); cal = fs == 9; te_in = fs == 8
    models = fit_ensemble(X[tr], y[tr], K=8)

    # calibration residuals
    pc, sc = predict_ensemble(models, X[cal]); res = np.abs(y[cal] - pc)
    nres = res / (sc + 1e-6)

    te = pd.read_csv(S669).reset_index(drop=True)
    emb_te = np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    llr_te = np.load(config.WORK + "/" + a.llr.replace(".npz", "_s669.npz"))["llr"][:, None]
    Xte = np.hstack([F.featurize_df(te).astype(np.float32), emb_te, llr_te]); yte = te.ddg.values

    out = {"tag": a.tag, "ensemble_K": len(models)}
    # point accuracy: single vs ensemble on S669
    p1 = anti(models[0], Xte); pe, se = predict_ensemble(models, Xte)
    out["s669_single_r"] = float(pearsonr(yte, p1)[0])
    out["s669_ensemble_r"] = float(pearsonr(yte, pe)[0])
    out["s669_ensemble_rmse"] = rmse(yte, pe)

    pin, sin = predict_ensemble(models, X[te_in]); yin = y[te_in]
    for alpha, nom in [(0.1, "90"), (0.2, "80")]:
        qs = np.quantile(res, 1 - alpha)                       # standard
        qn = np.quantile(nres, 1 - alpha)                      # normalized
        rec = {}
        # in-distribution (S2648 fold 8)
        rec["in_cov_std"] = coverage(yin, pin - qs, pin + qs)
        rec["in_cov_norm"] = coverage(yin, pin - qn * (sin + 1e-6), pin + qn * (sin + 1e-6))
        rec["in_width_std"] = float(2 * qs)
        rec["in_width_norm"] = float(np.mean(2 * qn * (sin + 1e-6)))
        # shift (S669)
        rec["s669_cov_std"] = coverage(yte, pe - qs, pe + qs)
        rec["s669_cov_norm"] = coverage(yte, pe - qn * (se + 1e-6), pe + qn * (se + 1e-6))
        rec["s669_width_std"] = float(2 * qs)
        rec["s669_width_norm"] = float(np.mean(2 * qn * (se + 1e-6)))
        out[f"nominal_{nom}"] = rec

    json.dump(out, open(config.RESULTS + f"/results_uncertainty_{a.tag}.json", "w"), indent=2)
    np.savez(config.WORK + f"/unc_{a.tag}.npz", yte=yte, pe=pe, se=se)
    print(f"[{a.tag}] ensemble S669 r: single={out['s669_single_r']:.3f} ensemble={out['s669_ensemble_r']:.3f}")
    for nom in ["90", "80"]:
        r = out[f"nominal_{nom}"]
        print(f"  nominal {nom}%:  in-dist cov std={r['in_cov_std']:.2f} norm={r['in_cov_norm']:.2f} | "
              f"S669 cov std={r['s669_cov_std']:.2f} norm={r['s669_cov_norm']:.2f} | "
              f"width std={r['s669_width_std']:.2f} norm~{r['s669_width_norm']:.2f} kcal/mol")
    print("saved", config.RESULTS + f"/results_uncertainty_{a.tag}.json")

if __name__ == "__main__": main()
