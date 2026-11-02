#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Train & evaluate the anti-symmetric ddG predictor.

- S2648: 10-fold CV using the predefined (protein-blind) splits shipped by ACDC-NN.
- Ssym : train on all S2648, test direct + inverse; report the symmetry bias.
- Ablation: a plain model (direct-only, no wrapper) to expose the destabilization bias
  that the anti-symmetric construction removes.
"""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

DATA = config.DATA + "/parsed/ddg_dataset.csv"
OUT = config.RESULTS + "/results.json"

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def xgb():
    return XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        min_child_weight=3, n_jobs=4, objective="reg:squarederror",
                        random_state=0)

def augment(X, y):
    """Add reverse examples (F -> -F, y -> -y) for a symmetric training set."""
    return np.vstack([X, -X]), np.concatenate([y, -y])

def anti_predict(model, X):
    """Exactly anti-symmetric prediction: (h(F) - h(-F)) / 2."""
    return (model.predict(X) - model.predict(-X)) / 2.0

def metrics(y, p):
    return dict(pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]),
                rmse=rmse(y, p), n=int(len(y)))

def main():
    df = pd.read_csv(DATA)
    s2648 = df[df.dataset == "S2648"].reset_index(drop=True)
    ssdir = df[df.dataset == "Ssym_dir"].reset_index(drop=True)
    ssinv = df[df.dataset == "Ssym_inv"].reset_index(drop=True)

    # ---- check the predefined folds are protein-disjoint (no leakage) ----
    fold_prot = {k: set(s2648[s2648.fold == k].protein) for k in range(10)}
    overlap = 0
    for a in range(10):
        for b in range(a + 1, 10):
            overlap += len(fold_prot[a] & fold_prot[b])
    print(f"S2648: {len(s2648)} mutations, {s2648.protein.nunique()} proteins, "
          f"cross-fold protein overlaps = {overlap} (0 => blind protein-level CV)")

    print("featurizing ...")
    Xall = F.featurize_df(s2648); yall = s2648.ddg.values
    folds = s2648.fold.values

    # ================= S2648 10-fold CV =================
    oof_anti = np.zeros(len(s2648)); oof_naive = np.zeros(len(s2648))
    for k in range(10):
        tr, te = folds != k, folds == k
        Xa, ya = augment(Xall[tr], yall[tr])
        m = xgb().fit(Xa, ya)
        oof_anti[te] = anti_predict(m, Xall[te])
        mn = xgb().fit(Xall[tr], yall[tr])          # ablation: direct-only, no wrapper
        oof_naive[te] = mn.predict(Xall[te])
    cv_anti = metrics(yall, oof_anti)
    cv_naive = metrics(yall, oof_naive)
    print("\n== S2648 10-fold CV (direct mutations) ==")
    print(f"  anti-symmetric : r={cv_anti['pearson']:.3f}  rho={cv_anti['spearman']:.3f}  rmse={cv_anti['rmse']:.3f}")
    print(f"  naive (ablate) : r={cv_naive['pearson']:.3f}  rho={cv_naive['spearman']:.3f}  rmse={cv_naive['rmse']:.3f}")

    # ================= Ssym (train on all S2648) =================
    Xa, ya = augment(Xall, yall)
    model = xgb().fit(Xa, ya)
    naive = xgb().fit(Xall, yall)

    Xdir = F.featurize_df(ssdir); ydir = ssdir.ddg.values
    Xinv = F.featurize_df(ssinv); yinv = ssinv.ddg.values
    pdir, pinv = anti_predict(model, Xdir), anti_predict(model, Xinv)
    npdir, npinv = naive.predict(Xdir), naive.predict(Xinv)

    ss = dict(
        anti=dict(dir=metrics(ydir, pdir), inv=metrics(yinv, pinv),
                  mean_pred_dir=float(pdir.mean()), mean_pred_inv=float(pinv.mean())),
        naive=dict(dir=metrics(ydir, npdir), inv=metrics(yinv, npinv),
                   mean_pred_dir=float(npdir.mean()), mean_pred_inv=float(npinv.mean())),
        exp=dict(mean_dir=float(ydir.mean()), mean_inv=float(yinv.mean())),
    )
    print("\n== Ssym (direct N={}, inverse N={}) ==".format(len(ydir), len(yinv)))
    print("  experimental mean ddG:   dir={:+.2f}  inv={:+.2f}  (a state function => opposite)".format(
        ydir.mean(), yinv.mean()))
    print("  ANTI-SYMMETRIC model:")
    print("    dir  r={:.3f} rmse={:.3f}   inv r={:.3f} rmse={:.3f}".format(
        ss['anti']['dir']['pearson'], ss['anti']['dir']['rmse'],
        ss['anti']['inv']['pearson'], ss['anti']['inv']['rmse']))
    print("    mean predicted:  dir={:+.2f}  inv={:+.2f}  (bias delta = {:+.2f})".format(
        pdir.mean(), pinv.mean(), pdir.mean() + pinv.mean()))
    print("  NAIVE (non-sym) model:")
    print("    dir  r={:.3f} rmse={:.3f}   inv r={:.3f} rmse={:.3f}".format(
        ss['naive']['dir']['pearson'], ss['naive']['dir']['rmse'],
        ss['naive']['inv']['pearson'], ss['naive']['inv']['rmse']))
    print("    mean predicted:  dir={:+.2f}  inv={:+.2f}  (bias delta = {:+.2f})".format(
        npdir.mean(), npinv.mean(), npdir.mean() + npinv.mean()))

    # ---- hard proof of the law: reverse each DIRECT mutation on the SAME structure ----
    Xrev = np.vstack([F.featurize(r.seq, int(r.pos0), r.mut, r.wt)   # swapped wt<->mut, same seq/pos
                      for r in ssdir.itertuples()])
    prev = anti_predict(model, Xrev)
    anti_err = float(np.max(np.abs(pdir + prev)))
    naive_rev = naive.predict(Xrev)
    naive_err = float(np.max(np.abs(npdir + naive_rev)))
    print("\n== reverse-on-same-structure (the thermodynamic law ddG_fwd = -ddG_rev) ==")
    print(f"  anti-symmetric max |pred_fwd + pred_rev| = {anti_err:.6f}  (0 = law satisfied exactly)")
    print(f"  naive          max |pred_fwd + pred_rev| = {naive_err:.3f}  (violation)")

    json.dump(dict(cv_anti=cv_anti, cv_naive=cv_naive, ssym=ss,
                   law_err_anti=anti_err, law_err_naive=naive_err,
                   n_features=len(F.FEATURE_NAMES)), open(OUT, "w"), indent=2)
    np.savez(config.WORK + "/preds.npz",
             y_cv=yall, p_cv_anti=oof_anti, p_cv_naive=oof_naive,
             ydir=ydir, yinv=yinv, pdir=pdir, pinv=pinv, npdir=npdir, npinv=npinv)
    print("\nsaved", OUT)

if __name__ == "__main__":
    main()
