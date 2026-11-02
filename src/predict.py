#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Predict ddG of folding for point mutations (anti-symmetric biophysical model).

Usage:
    python predict.py <SEQUENCE> <MUT> [<MUT> ...]
    python predict.py --fasta file.fasta <MUT> [...]

A mutation is <wt><pos1based><mut>, e.g. M1A, V47I.
Sign convention of the S2648/Ssym labels: negative ddG = DEstabilizing, positive = stabilizing.
Prints the prediction and its exact reverse to demonstrate the thermodynamic law.
"""
import sys, numpy as np
from xgboost import XGBRegressor
import features as F

MODEL=config.MODELS + "/model_bio.json"

def load():
    m=XGBRegressor(); m.load_model(MODEL); return m
def anti(m,X): return (m.predict(X)-m.predict(-X))/2.0

def main(argv):
    if len(argv)<2:
        print("Usage: python predict.py <SEQUENCE|--fasta file> <MUT> [<MUT> ...]\n"
              "  MUT = <wt><pos1based><mut>, e.g. M1A V47I\n"
              "  sign: negative ddG = destabilizing, positive = stabilizing"); return
    if argv[0]=="--fasta":
        seq="".join(l.strip() for l in open(argv[1]) if not l.startswith(">"))
        muts=argv[2:]
    else:
        seq=argv[0].strip().upper(); muts=argv[1:]
    seq="".join(c for c in seq.upper() if c in F.AAs)
    m=load()
    print(f"sequence length {len(seq)}\n")
    print(f"{'mutation':>10}  {'ddG_pred':>9}  {'reverse':>9}  interpretation")
    print("-"*58)
    for mut in muts:
        wt,mt=mut[0],mut[-1]; pos=int(mut[1:-1])
        if pos<1 or pos>len(seq):
            print(f"{mut:>10}  position out of range"); continue
        if seq[pos-1]!=wt:
            print(f"{mut:>10}  wt mismatch: sequence has {seq[pos-1]} at {pos}"); continue
        Xf=F.featurize(seq,pos-1,wt,mt)[None,:]
        Xr=F.featurize(seq[:pos-1]+mt+seq[pos:],pos-1,mt,wt)[None,:]
        d=float(anti(m,Xf)[0]); dr=float(anti(m,Xr)[0])
        tag="stabilizing" if d>0.5 else ("destabilizing" if d<-0.5 else "~neutral")
        print(f"{mut:>10}  {d:>+9.2f}  {dr:>+9.2f}  {tag}")
    print("\n(reverse column is the model's prediction for the back-mutation; it equals")
    print(" -ddG_pred exactly, as a folding free-energy difference must.)")

if __name__=="__main__": main(sys.argv[1:])
