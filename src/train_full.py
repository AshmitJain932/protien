#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Final model: biophysical + ESM2 embedding deltas + ESM2 zero-shot LLR, anti-symmetric.
Reports S2648 protein-blind CV and the S669 blind test for each feature set, plus the
training-free zero-shot LLR baseline for context."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

DDG=config.DATA + "/parsed/ddg_dataset.csv"
S669=config.DATA + "/parsed/s669_dataset.csv"

def xgb(): return XGBRegressor(n_estimators=700,max_depth=4,learning_rate=0.03,subsample=0.8,
    colsample_bytree=0.6,reg_lambda=2.0,min_child_weight=4,n_jobs=4,random_state=0)
def augment(X,y): return np.vstack([X,-X]), np.concatenate([y,-y])
def anti(m,X): return (m.predict(X)-m.predict(-X))/2.0
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def met(y,p): return dict(pearson=float(pearsonr(y,p)[0]),spearman=float(spearmanr(y,p)[0]),rmse=rmse(y,p),n=int(len(y)))

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--llr", default="llr.npz"); a=ap.parse_args()
    tr=pd.read_csv(DDG); m2=(tr.dataset=="S2648").values
    emb=np.load(config.WORK + "/esm_delta.npz")["delta"]
    llr=np.load(config.WORK + "/" + a.llr)["llr"][:,None]
    Xbio=F.featurize_df(tr).astype(np.float32)
    s=tr[m2].reset_index(drop=True); ys=s.ddg.values; fs=s.fold.values
    bio_s,emb_s,llr_s=Xbio[m2],emb[m2],llr[m2]

    te=pd.read_csv(S669).reset_index(drop=True); yte=te.ddg.values
    Xbio_te=F.featurize_df(te).astype(np.float32)
    emb_te=np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    llr_te=np.load(config.WORK + "/" + a.llr.replace(".npz","_s669.npz"))["llr"][:,None]

    sets={
      "bio":                 (bio_s, Xbio_te),
      "bio+esm":             (np.hstack([bio_s,emb_s]), np.hstack([Xbio_te,emb_te])),
      "bio+esm+llr (FULL)":  (np.hstack([bio_s,emb_s,llr_s]), np.hstack([Xbio_te,emb_te,llr_te])),
    }
    res={}
    for name,(Xs,Xte) in sets.items():
        oof=np.zeros(len(ys))
        for k in range(10):
            trm=fs!=k; tem=fs==k
            Xa,ya=augment(Xs[trm],ys[trm]); mdl=xgb().fit(Xa,ya); oof[tem]=anti(mdl,Xs[tem])
        cv=met(ys,oof)
        Xa,ya=augment(Xs,ys); full=xgb().fit(Xa,ya); pte=anti(full,Xte)
        blind=met(yte,pte); blind["mean_pred"]=float(pte.mean())
        res[name]=dict(cv=cv,s669=blind)
        print(f"{name:22s} | S2648 CV r={cv['pearson']:.3f} rho={cv['spearman']:.3f} rmse={cv['rmse']:.2f} "
              f"| S669 r={blind['pearson']:.3f} rho={blind['spearman']:.3f} rmse={blind['rmse']:.2f}")

    # training-free zero-shot LLR baseline (no model at all)
    zs_cv=met(ys, llr_s[:,0]); zs_te=met(yte, llr_te[:,0])
    res["zeroshot_LLR_only"]=dict(cv=zs_cv,s669=zs_te)
    print(f"{'zero-shot LLR (no train)':22s} | S2648 r={zs_cv['pearson']:.3f} | S669 r={zs_te['pearson']:.3f}  (training-free)")

    # save final deployable full model + predictions/figure arrays
    Xs=sets["bio+esm+llr (FULL)"][0]; Xa,ya=augment(Xs,ys)
    xgb().fit(Xa,ya).save_model(config.MODELS + "/model_full.json")
    Xte=sets["bio+esm+llr (FULL)"][1]
    oof=np.zeros(len(ys))
    for k in range(10):
        trm=fs!=k; tem=fs==k; Xa,ya=augment(Xs[trm],ys[trm]); mdl=xgb().fit(Xa,ya); oof[tem]=anti(mdl,Xs[tem])
    full=xgb().fit(*augment(Xs,ys))
    np.savez(config.WORK + "/preds_full.npz",y_cv=ys,p_cv=oof,y669=yte,p669=anti(full,Xte),llr669=llr_te[:,0])
    json.dump(res,open(config.RESULTS + "/results_final.json","w"),indent=2)
    print("\nsaved results_final.json, preds_full.npz, model_full.json")

if __name__=="__main__": main()
