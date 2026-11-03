#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Train the anti-symmetric model on the mega-scale dataset (~270k mutations, 298 domains) with
biophysical + ESM zero-shot LLR features, and evaluate:
  (1) the clean held-out mega-scale test split (protein-blind) — overall and per-protein ranking;
  (2) transfer to the S669 blind set (trained on mega-scale, tested on S669).
"""
import json, time, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

MEGA = config.DATA + "/parsed/megascale.csv"
S669 = config.DATA + "/parsed/s669_dataset.csv"

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    t0 = time.time()
    df = pd.read_csv(MEGA)
    print(f"featurizing {len(df)} mega-scale mutations ...", flush=True)
    bio = F.featurize_df(df).astype(np.float32)
    llr = np.load(config.WORK + "/llr_mega.npz")["llr"][:, None]
    use_emb = _os.path.exists(config.WORK + "/esm_delta_mega.npz")
    feats = [bio]
    if use_emb:
        feats.append(np.load(config.WORK + "/esm_delta_mega.npz")["delta"]); print("using mega ESM embeddings")
    feats.append(llr)
    X = np.hstack(feats); y = df.value.values
    tr = df.split.isin(["train", "val"]).values; te = (df.split == "test").values
    print(f"  featurized in {time.time()-t0:.0f}s; train {tr.sum()}, test {te.sum()}", flush=True)

    model = XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=8, n_jobs=4,
                         random_state=0).fit(*augment(X[tr], y[tr]))
    pte = anti(model, X[te]); yte = y[te]
    overall = dict(pearson=float(pearsonr(yte, pte)[0]), spearman=float(spearmanr(yte, pte)[0]),
                   rmse=rmse(yte, pte), n=int(te.sum()))
    # per-protein ranking (engineering-relevant)
    dte = df[te].copy(); dte["pred"] = pte
    per = [spearmanr(g.value, g.pred)[0] for _, g in dte.groupby("protein") if len(g) >= 10]
    per = [x for x in per if x == x]
    perprot = dict(mean_spearman=float(np.mean(per)), median_spearman=float(np.median(per)), n_proteins=len(per))

    # transfer to S669 (opposite ddG sign convention -> compare by magnitude)
    s = pd.read_csv(S669).reset_index(drop=True)
    sfeats = [F.featurize_df(s).astype(np.float32)]
    if use_emb:
        sfeats.append(np.load(config.WORK + "/esm_delta_s669.npz")["delta"])
    sfeats.append(np.load(config.WORK + "/llr_s669.npz")["llr"][:, None])
    Xs = np.hstack(sfeats)
    ps = anti(model, Xs); r_s669 = pearsonr(s.ddg.values, ps)[0]
    transfer = dict(pearson_signed=float(r_s669), pearson_abs=float(abs(r_s669)), n=len(s))

    # --- save the trained model + reproducibility card ---
    _os.makedirs(config.MODELS, exist_ok=True)
    blocks = [["biophysical", int(bio.shape[1])]]
    if use_emb: blocks.append(["esm_delta", int(np.load(config.WORK + "/esm_delta_mega.npz")["delta"].shape[1])])
    blocks.append(["llr", 1])
    model.save_model(_os.path.join(config.MODELS, "model_mega.json"))
    json.dump(dict(model_file="model_mega.json", feature_dim=int(X.shape[1]), feature_blocks=blocks,
                   anti_symmetric=True, n_train_mutations=int(tr.sum()), used_esm_embeddings=bool(use_emb),
                   inference="pred = (m.predict(F) - m.predict(-F)) / 2 ; concat feature_blocks in order.",
                   metrics=dict(mega_test=overall, mega_per_protein=perprot)),
              open(_os.path.join(config.MODELS, "model_cards_mega.json"), "w"), indent=2)

    out = dict(mega_test=overall, mega_per_protein=perprot, s669_transfer=transfer)
    json.dump(out, open(config.RESULTS + "/results_megascale.json", "w"), indent=2)
    print("\n=== mega-scale held-out TEST (protein-blind, 28 domains) ===")
    print(f"  overall: r={overall['pearson']:.3f}  rho={overall['spearman']:.3f}  rmse={overall['rmse']:.3f}")
    print(f"  per-protein ranking: mean Spearman={perprot['mean_spearman']:.3f} "
          f"median={perprot['median_spearman']:.3f} over {perprot['n_proteins']} domains")
    print(f"\n=== transfer to S669 (trained on mega-scale) ===")
    print(f"  |r| = {transfer['pearson_abs']:.3f}  (signed {transfer['pearson_signed']:+.3f}; "
          f"sign flips due to opposite ddG convention)")
    print("saved results_megascale.json")

if __name__ == "__main__":
    main()
