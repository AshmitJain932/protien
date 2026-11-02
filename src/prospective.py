#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""In-silico 'prospective design' test (the honest stand-in for wet-lab #7).

For each protein in the S669 BLIND set (never seen in training), rank its measured mutations by
the model's predicted stabilization and ask a design-relevant question: if an engineer followed the
model's top picks, how often would they actually get a stabilizing mutation? We report precision
among predicted-stabilizing vs. the base rate (enrichment), and top-k hit rates. This mirrors
'predict -> make -> measure' using held-out experimental values in place of the wet lab.
"""
import json, numpy as np, pandas as pd
from xgboost import XGBRegressor
import features as F

DDG = config.DATA + "/parsed/ddg_dataset.csv"
S669 = config.DATA + "/parsed/s669_dataset.csv"

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0

def main(llr="llr.npz"):
    df = pd.read_csv(DDG); m2 = (df.dataset == "S2648").values
    emb = np.load(config.WORK + "/esm_delta.npz")["delta"]
    ll = np.load(config.WORK + "/" + llr)["llr"][:, None]
    Xtr = np.hstack([F.featurize_df(df).astype(np.float32), emb, ll])[m2]
    ytr = df[m2].ddg.values
    m = XGBRegressor(n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=4, n_jobs=4,
                     random_state=0).fit(*augment(Xtr, ytr))

    te = pd.read_csv(S669).reset_index(drop=True)
    emb_te = np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    ll_te = np.load(config.WORK + "/" + llr.replace(".npz", "_s669.npz"))["llr"][:, None]
    Xte = np.hstack([F.featurize_df(te).astype(np.float32), emb_te, ll_te])
    pred = anti(m, Xte); te["pred"] = pred

    STAB = 0.0  # ddG > 0 = stabilizing (dataset convention)
    base_rate = float((te.ddg > STAB).mean())
    # precision among predicted-stabilizing
    psel = te.pred > STAB
    prec = float((te.ddg[psel] > STAB).mean()) if psel.sum() else float("nan")
    enrich = prec / base_rate if base_rate else float("nan")

    # per-protein top-1 / top-3 hit rate (engineer tries the model's best few on each target)
    hits1 = hits3 = n = 0
    for _, g in te.groupby("protein"):
        if len(g) < 3: continue
        n += 1
        top = g.sort_values("pred", ascending=False)
        hits1 += int(top.iloc[0].ddg > STAB)
        hits3 += int((top.head(3).ddg > STAB).any())
    out = dict(n_blind=len(te), base_rate_stabilizing=base_rate,
               precision_pred_stabilizing=prec, enrichment_vs_base=enrich,
               proteins_scored=n, top1_hit_rate=hits1 / n, top3_hit_rate=hits3 / n)
    print("=== In-silico prospective design on S669 (blind) ===")
    print(f"  base rate of stabilizing mutations      : {base_rate:.1%}")
    print(f"  precision among predicted-stabilizing    : {prec:.1%}  (enrichment {enrich:.2f}x)")
    print(f"  per-protein top-1 pick is stabilizing    : {out['top1_hit_rate']:.1%}")
    print(f"  per-protein top-3 contains a stabilizer  : {out['top3_hit_rate']:.1%}  (over {n} proteins)")
    json.dump(out, open(config.RESULTS + "/results_prospective.json", "w"), indent=2)
    print("saved", config.RESULTS + "/results_prospective.json")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "llr.npz")
