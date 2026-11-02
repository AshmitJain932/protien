#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Stage 2: fuse biophysical deltas with ESM2 embedding deltas, keep exact anti-symmetry.
Ablation over feature sets {bio, esm, bio+esm}; same protein-blind CV + Ssym protocol."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
import features as F

DATA = config.DATA + "/parsed/ddg_dataset.csv"

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def xgb(): return XGBRegressor(n_estimators=700,max_depth=4,learning_rate=0.03,subsample=0.8,
    colsample_bytree=0.6,reg_lambda=2.0,min_child_weight=4,n_jobs=4,
    objective="reg:squarederror",random_state=0)
def augment(X,y): return np.vstack([X,-X]), np.concatenate([y,-y])
def anti(m,X): return (m.predict(X)-m.predict(-X))/2.0
def met(y,p): return dict(pearson=float(pearsonr(y,p)[0]),spearman=float(spearmanr(y,p)[0]),rmse=rmse(y,p),n=int(len(y)))

def main():
    df=pd.read_csv(DATA)
    ed=np.load(config.WORK + "/esm_delta.npz")["delta"]   # aligned to csv row order
    assert len(ed)==len(df), (len(ed),len(df))
    df=df.reset_index(drop=True)
    is2648=(df.dataset=="S2648").values
    isdir=(df.dataset=="Ssym_dir").values
    isinv=(df.dataset=="Ssym_inv").values

    Xbio=F.featurize_df(df).astype(np.float32)
    Xesm=ed
    sets={"bio":Xbio,"esm":Xesm,"bio+esm":np.hstack([Xbio,Xesm])}

    y=df.ddg.values; folds=df.fold.values
    out={}
    for name,Xall in sets.items():
        # ---- S2648 protein-blind 10-fold CV ----
        oof=np.zeros(is2648.sum())
        Xs=Xall[is2648]; ys=y[is2648]; fs=folds[is2648]
        idx=np.arange(len(ys))
        for k in range(10):
            tr=fs!=k; te=fs==k
            Xa,ya=augment(Xs[tr],ys[tr])
            m=xgb().fit(Xa,ya)
            oof[te]=anti(m,Xs[te])
        cv=met(ys,oof)
        # ---- Ssym: train on all S2648, test dir/inv ----
        Xa,ya=augment(Xs,ys); model=xgb().fit(Xa,ya)
        pdir=anti(model,Xall[isdir]); pinv=anti(model,Xall[isinv])
        ydir=y[isdir]; yinv=y[isinv]
        ss=dict(dir=met(ydir,pdir),inv=met(yinv,pinv),
                mean_pred_dir=float(pdir.mean()),mean_pred_inv=float(pinv.mean()),
                bias=float(pdir.mean()+pinv.mean()))
        out[name]=dict(cv=cv,ssym=ss)
        print(f"\n### feature set: {name}  (dim={Xall.shape[1]})")
        print(f"  S2648 CV : r={cv['pearson']:.3f} rho={cv['spearman']:.3f} rmse={cv['rmse']:.3f}")
        print(f"  Ssym dir : r={ss['dir']['pearson']:.3f} rmse={ss['dir']['rmse']:.3f}")
        print(f"  Ssym inv : r={ss['inv']['pearson']:.3f} rmse={ss['inv']['rmse']:.3f}")
        print(f"  Ssym bias δ={ss['bias']:+.3f}  (0 = perfectly anti-symmetric)")

    json.dump(out,open(config.RESULTS + "/results_esm.json","w"),indent=2)
    # save combined preds for plotting
    name="bio+esm"; Xall=sets[name]
    Xs=Xall[is2648]; ys=y[is2648]; fs=folds[is2648]; oof=np.zeros(len(ys))
    for k in range(10):
        tr=fs!=k; te=fs==k; Xa,ya=augment(Xs[tr],ys[tr]); m=xgb().fit(Xa,ya); oof[te]=anti(m,Xs[te])
    Xa,ya=augment(Xs,ys); model=xgb().fit(Xa,ya)
    np.savez(config.WORK + "/preds_esm.npz",y_cv=ys,p_cv=oof,
             ydir=y[isdir],yinv=y[isinv],pdir=anti(model,Xall[isdir]),pinv=anti(model,Xall[isinv]))
    print("\nsaved results_esm.json + preds_esm.npz")

if __name__=="__main__": main()
