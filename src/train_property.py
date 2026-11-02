#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Property-agnostic anti-symmetric trainer. Reads any parsed CSV with columns
seq, pos0, wt, mut, value and runs protein-blind cross-validation with the exact anti-symmetric
wrapper. Swap the `value` column for any mutational property (stability, dTm, solubility change,
DMS fitness effect) and the same recipe applies. Demonstrated here on FireProtDB stability."""
import argparse, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
import features as F

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA + "/parsed/fireprot_ddg.csv")
    ap.add_argument("--label", default="FireProt stability (ddG)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(a.data).reset_index(drop=True)
    X = F.featurize_df(df).astype(np.float32); y = df.value.values
    groups = df.protein.values
    gkf = GroupKFold(n_splits=a.folds)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        Xa, ya = augment(X[tr], y[tr])
        m = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3, n_jobs=4,
                         random_state=0).fit(Xa, ya)
        oof[te] = anti(m, X[te])
    r = pearsonr(y, oof)[0]; rho = spearmanr(y, oof)[0]
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    print(f"{a.label}: N={len(y)} proteins={df.protein.nunique()} "
          f"| protein-blind {a.folds}-fold CV  r={r:.3f}  rho={rho:.3f}  rmse={rmse:.3f}")
    print("(anti-symmetric by construction; biophysical features only)")

if __name__ == "__main__":
    main()
