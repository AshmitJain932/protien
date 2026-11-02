#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Blind-test evaluation on S669 (zero sequence overlap with S2648 training).
Trains the anti-symmetric bio+ESM model on all S2648, predicts S669, reports metrics,
and contrasts with a naive (non-anti-symmetric) model to expose bias."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import features as F
from xgboost import XGBRegressor

DDG=config.DATA + "/parsed/ddg_dataset.csv"
S669=config.DATA + "/parsed/s669_dataset.csv"

def xgb(): return XGBRegressor(n_estimators=700,max_depth=4,learning_rate=0.03,subsample=0.8,
    colsample_bytree=0.6,reg_lambda=2.0,min_child_weight=4,n_jobs=4,random_state=0)
def augment(X,y): return np.vstack([X,-X]), np.concatenate([y,-y])
def anti(m,X): return (m.predict(X)-m.predict(-X))/2.0
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main():
    tr=pd.read_csv(DDG); s=tr[tr.dataset=="S2648"].reset_index(drop=True)
    ed=np.load(config.WORK + "/esm_delta.npz")["delta"]
    ed_s2648=ed[(tr.dataset=="S2648").values]
    Xtr=np.hstack([F.featurize_df(s).astype(np.float32), ed_s2648]); ytr=s.ddg.values

    te=pd.read_csv(S669).reset_index(drop=True)
    ed669=np.load(config.WORK + "/esm_delta_s669.npz")["delta"]
    Xte=np.hstack([F.featurize_df(te).astype(np.float32), ed669]); yte=te.ddg.values

    # anti-symmetric model
    Xa,ya=augment(Xtr,ytr); model=xgb().fit(Xa,ya); pred=anti(model,Xte)
    # naive model (direct only, no wrapper)
    naive=xgb().fit(Xtr,ytr); predn=naive.predict(Xte)

    out=dict(
      anti=dict(pearson=float(pearsonr(yte,pred)[0]),spearman=float(spearmanr(yte,pred)[0]),
                rmse=rmse(yte,pred),mean_pred=float(pred.mean()),n=len(yte)),
      naive=dict(pearson=float(pearsonr(yte,predn)[0]),spearman=float(spearmanr(yte,predn)[0]),
                 rmse=rmse(yte,predn),mean_pred=float(predn.mean()),n=len(yte)),
      exp_mean=float(yte.mean()))
    print("=== S669 blind test (N={}, {} proteins, 0 overlap with training) ===".format(
        len(yte), te.protein.nunique()))
    print("  experimental mean ddG = {:+.2f}".format(yte.mean()))
    print("  ANTI-SYMMETRIC : r={:.3f}  rho={:.3f}  rmse={:.3f}  mean_pred={:+.2f}".format(
        out['anti']['pearson'],out['anti']['spearman'],out['anti']['rmse'],out['anti']['mean_pred']))
    print("  naive          : r={:.3f}  rho={:.3f}  rmse={:.3f}  mean_pred={:+.2f}".format(
        out['naive']['pearson'],out['naive']['spearman'],out['naive']['rmse'],out['naive']['mean_pred']))
    json.dump(out,open(config.RESULTS + "/results_s669.json","w"),indent=2)
    np.savez(config.WORK + "/preds_s669.npz",y=yte,pred=pred,predn=predn)
    print("saved results_s669.json + preds_s669.npz")

if __name__=="__main__": main()
