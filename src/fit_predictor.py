#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Fit the deployable biophysical anti-symmetric predictor on all of S2648 and save it.
(The biophysical model needs no GPU/torch at inference, so it powers a fast CLI.)"""
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import features as F

DATA=config.DATA + "/parsed/ddg_dataset.csv"
def main():
    df=pd.read_csv(DATA); s=df[df.dataset=="S2648"].reset_index(drop=True)
    X=F.featurize_df(s).astype(np.float32); y=s.ddg.values
    Xa=np.vstack([X,-X]); ya=np.concatenate([y,-y])
    m=XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,reg_lambda=1.0,min_child_weight=3,n_jobs=4,random_state=0)
    m.fit(Xa,ya)
    m.save_model(config.MODELS + "/model_bio.json")
    print("saved model_bio.json  (trained on %d S2648 mutations, anti-symmetric wrapper applied at predict time)"%len(s))
if __name__=="__main__": main()
