#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Structure-aware model: bio + ESM2 embeddings + zero-shot LLR + structure (SASA/contacts),
all anti-symmetric. Reports S2648 protein-blind CV and S669 blind, with/without structure."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

DDG = config.DATA + "/parsed/ddg_dataset.csv"
S669 = config.DATA + "/parsed/s669_dataset.csv"

def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
XGB_PARAMS = dict(n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=4, n_jobs=4, random_state=0)
def xgb(): return XGBRegressor(**XGB_PARAMS)
def met(y, p): return dict(pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]), rmse=rmse(y, p))
def slug(name): return (name.replace(" ", "_").replace("+", "plus").replace("(", "")
                        .replace(")", "").replace("__", "_").strip("_"))

def main(llr="llr.npz"):
    tr = pd.read_csv(DDG); m2 = (tr.dataset == "S2648").values
    bio = F.featurize_df(tr).astype(np.float32)
    emb = np.load(config.WORK + "/esm_delta.npz")["delta"]
    ll = np.load(config.WORK + "/" + llr)["llr"][:, None]
    st = np.load(config.WORK + "/struct_inter.npz")["X"]
    def opt(path, col=None):
        try:
            d = np.load(config.WORK + "/" + path)
            v = d["llr"] if "llr" in d else d["X"]
            return v[:, None] if v.ndim == 1 else v
        except Exception:
            return None
    mpnn = opt("mpnn_llr.npz")
    s = tr[m2].reset_index(drop=True); y = s.ddg.values; fs = s.fold.values

    te = pd.read_csv(S669).reset_index(drop=True); yte = te.ddg.values
    bio_te = F.featurize_df(te).astype(np.float32)
    emb_te = np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    ll_te = np.load(config.WORK + "/" + llr.replace(".npz", "_s669.npz"))["llr"][:, None]
    st_te = np.load(config.WORK + "/struct_inter_s669.npz")["X"]
    mpnn_te = opt("mpnn_llr_s669.npz")

    base = [bio[m2], emb[m2], ll[m2]]; base_te = [bio_te, emb_te, ll_te]
    sets = {
        "seq (bio+esm+llr)":    (np.hstack(base),            np.hstack(base_te)),
        "+ structure":          (np.hstack(base + [st[m2]]), np.hstack(base_te + [st_te])),
    }
    if mpnn is not None and mpnn_te is not None:
        sets["+ structure + ProteinMPNN"] = (np.hstack(base + [st[m2], mpnn[m2]]),
                                             np.hstack(base_te + [st_te, mpnn_te]))
    # feature-block layout per set (ordered) so a saved model can be reassembled at inference
    bd, ed, sd = bio.shape[1], emb.shape[1], st.shape[1]
    blocks = {
        "seq (bio+esm+llr)":           [["biophysical", bd], ["esm_delta", ed], ["llr", 1]],
        "+ structure":                 [["biophysical", bd], ["esm_delta", ed], ["llr", 1], ["struct_inter", sd]],
        "+ structure + ProteinMPNN":   [["biophysical", bd], ["esm_delta", ed], ["llr", 1], ["struct_inter", sd], ["mpnn_llr", 1]],
    }
    _os.makedirs(config.MODELS, exist_ok=True)
    res, cards = {}, {}
    for name, (X, Xte) in sets.items():
        oof = np.zeros(len(y))
        for k in range(10):
            trm = fs != k; tem = fs == k
            oof[tem] = anti(xgb().fit(*augment(X[trm], y[trm])), X[tem])
        cv = met(y, oof)
        full = xgb().fit(*augment(X, y)); blind = met(yte, anti(full, Xte))
        res[name] = dict(cv=cv, s669=blind)
        # --- save the trained model + a reproducibility card ---
        mfile = f"model_struct__{slug(name)}.json"
        full.save_model(_os.path.join(config.MODELS, mfile))
        cards[name] = dict(
            model_file=mfile, feature_dim=int(X.shape[1]), feature_blocks=blocks[name],
            xgb_params=XGB_PARAMS, n_train_mutations=int(len(y)), llr_source=llr,
            anti_symmetric=True,
            inference="pred = (m.predict(F) - m.predict(-F)) / 2 ; build F by concatenating the "
                      "feature_blocks in order using esm_features.py / zeroshot.py / "
                      "struct_features.py / proteinmpnn_features.py with the SAME model/settings.",
            metrics=res[name])
        print(f"{name:22s} | S2648 CV r={cv['pearson']:.3f} rmse={cv['rmse']:.2f} "
              f"| S669 r={blind['pearson']:.3f} rho={blind['spearman']:.3f} rmse={blind['rmse']:.2f} "
              f"| saved {mfile}")
    json.dump(res, open(config.RESULTS + "/results_struct.json", "w"), indent=2)
    json.dump(cards, open(_os.path.join(config.MODELS, "model_cards_struct.json"), "w"), indent=2)
    print("saved results_struct.json + model checkpoints + model_cards_struct.json")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "llr.npz")
