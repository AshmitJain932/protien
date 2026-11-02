#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Build & verify the S669 blind-test set (PDB AUTHOR numbering) and check training overlap."""
import sys, csv, pandas as pd
sys.path.insert(0, config.WORK + "")
from build_dataset import THREE2ONE

PDB=config.DATA + "/pdbs"
SRC=config.DATA + ""
S669=config.DATA + "/ThermoMPNN_Kuhlman-Lab/data_all/testing/s669_clean_dir.csv"
OUT=f"{SRC}/parsed/s669_dataset.csv"

def seq_and_authormap(pdb,chain):
    """observed-order sequence + {author_resseq -> index in that sequence} (blank icode only)."""
    seq=[]; author={}; seen=set(); started=False; inmodel=True
    for line in open(f"{PDB}/{pdb.upper()}.pdb",errors="ignore"):
        rec=line[:6].strip()
        if rec=="MODEL":
            if started: inmodel=False
            else: started=True; inmodel=True
            continue
        if rec in ("ATOM","HETATM") and inmodel and line[12:16].strip()=="CA" and line[21]==chain:
            if line[16] not in (' ','A'): continue
            one=THREE2ONE.get(line[17:20].strip(),'X')
            if one in ('U','O'): one='X'
            try: rs=int(line[22:26])
            except ValueError: continue
            ic=line[26].strip(); k=(rs,ic)
            if k in seen: continue
            seen.add(k); idx=len(seq); seq.append(one)
            if ic=="" and rs not in author: author[rs]=idx
    return "".join(seq), author

def build():
    df=pd.read_csv(S669); rows=[]; matched=0; seqs={}
    for r in df.itertuples():
        pdb,chain=r.PDB[:4],r.PDB[4]; wt,mt=r.MUT[0],r.MUT[-1]
        pos=int(r.MUT[1:-1]); ddg=float(r.DDG)
        s,au=seq_and_authormap(pdb,chain)
        if pos in au and s[au[pos]]==wt:
            i=au[pos]; matched+=1; seqs[f"{pdb}{chain}"]=s
            rows.append(dict(dataset="S669",protein=f"{pdb}{chain}",pdb=pdb,chain=chain,
                             pos0=i,wt=wt,mut=mt,ddg=ddg,seq=s,fold=-3))
    with open(OUT,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["dataset","protein","pdb","chain","pos0","wt","mut","ddg","seq","fold"])
        w.writeheader(); w.writerows(rows)
    print(f"S669: matched {matched}/{len(df)} ({100*matched/len(df):.1f}%)  | unique proteins {len(seqs)}")

    tr=pd.read_csv(f"{SRC}/parsed/ddg_dataset.csv")
    tr_seqs=set(tr[tr.dataset=="S2648"].seq); s669_seqs=set(seqs.values())
    exact=sum(1 for s in s669_seqs if s in tr_seqs)
    def kmers(s,k=6): return set(s[i:i+k] for i in range(len(s)-k+1))
    trk=[kmers(s) for s in tr_seqs]
    near=sum(1 for s in s669_seqs if max((len(kmers(s)&t)/max(len(kmers(s)|t),1) for t in trk),default=0)>=0.9)
    print(f"overlap with S2648 training: exact-sequence {exact}/{len(s669_seqs)}, >=0.9 6-mer Jaccard {near}/{len(s669_seqs)}")

if __name__=="__main__": build()
