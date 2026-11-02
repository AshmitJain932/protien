#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Benchmark figure for the anti-symmetric ddG predictor."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

d = np.load(config.WORK + "/preds.npz")
R = json.load(open(config.RESULTS + "/results.json"))

INK="#1b2230"; MUT="#5b6475"; GRID="#e6eaf1"
C_ANTI="#4f46e5"; C_DIR="#0d8a89"; C_INV="#c47d1c"; C_NAIVE="#d13b3b"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUT,
    "axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK,
    "axes.titlesize":11,"figure.dpi":130})

def scatter(ax,x,y,color,label=None):
    ax.scatter(x,y,s=10,alpha=0.45,color=color,edgecolors="none",label=label)

def diag(ax,lo,hi):
    ax.plot([lo,hi],[lo,hi],"--",color=MUT,lw=1,zorder=0)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal")
    ax.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)

fig,ax=plt.subplots(2,2,figsize=(10,9.4))
fig.suptitle("Anti-symmetric sequence-only ΔΔG predictor — benchmark",fontsize=14,fontweight="bold",y=0.98)

# (0,0) S2648 CV anti
a=ax[0,0]; y,p=d["y_cv"],d["p_cv_anti"]; r=pearsonr(y,p)[0]; rho=spearmanr(y,p)[0]
rmse=np.sqrt(np.mean((y-p)**2))
scatter(a,y,p,C_ANTI); diag(a,-7,7)
a.set_title(f"S2648 · protein-blind 10-fold CV\nPearson r={r:.3f}  ρ={rho:.3f}  RMSE={rmse:.2f} kcal/mol")
a.set_xlabel("experimental ΔΔG"); a.set_ylabel("predicted ΔΔG")

# (0,1) Ssym anti (dir + inv overlaid)
a=ax[0,1]
scatter(a,d["ydir"],d["pdir"],C_DIR,f"direct  r={pearsonr(d['ydir'],d['pdir'])[0]:.2f}")
scatter(a,d["yinv"],d["pinv"],C_INV,f"inverse r={pearsonr(d['yinv'],d['pinv'])[0]:.2f}")
diag(a,-7,7); a.legend(loc="upper left",frameon=False,fontsize=9)
bias=R["ssym"]["anti"]["mean_pred_dir"]+R["ssym"]["anti"]["mean_pred_inv"]
a.set_title(f"Ssym · ANTI-SYMMETRIC model\nbalanced both directions · bias δ={bias:+.2f} kcal/mol")
a.set_xlabel("experimental ΔΔG"); a.set_ylabel("predicted ΔΔG")

# (1,0) Ssym naive (dir + inv overlaid) -> inverse collapses
a=ax[1,0]
scatter(a,d["ydir"],d["npdir"],C_DIR,f"direct  r={pearsonr(d['ydir'],d['npdir'])[0]:.2f}")
scatter(a,d["yinv"],d["npinv"],C_NAIVE,f"inverse r={pearsonr(d['yinv'],d['npinv'])[0]:.2f}")
diag(a,-7,7); a.legend(loc="upper left",frameon=False,fontsize=9)
biasn=R["ssym"]["naive"]["mean_pred_dir"]+R["ssym"]["naive"]["mean_pred_inv"]
a.set_title(f"Ssym · NAIVE model (no symmetry)\ninverse collapses · bias δ={biasn:+.2f} kcal/mol")
a.set_xlabel("experimental ΔΔG"); a.set_ylabel("predicted ΔΔG")

# (1,1) grouped bars r_dir vs r_inv
a=ax[1,1]
groups=["direct r","inverse r"]; x=np.arange(2); w=0.36
anti=[R["ssym"]["anti"]["dir"]["pearson"],R["ssym"]["anti"]["inv"]["pearson"]]
naive=[R["ssym"]["naive"]["dir"]["pearson"],R["ssym"]["naive"]["inv"]["pearson"]]
b1=a.bar(x-w/2,naive,w,color=C_NAIVE,label="naive (standard ML)")
b2=a.bar(x+w/2,anti,w,color=C_ANTI,label="anti-symmetric (ours)")
for b in list(b1)+list(b2):
    a.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{b.get_height():.2f}",
           ha="center",fontsize=9,color=INK)
a.set_xticks(x); a.set_xticklabels(groups); a.set_ylim(0,0.82)
a.grid(True,axis="y",color=GRID,lw=0.8); a.set_axisbelow(True)
a.legend(frameon=False,fontsize=9,loc="upper right")
a.set_title("Ssym correlation: the naive model looks great on\ndirect mutations but fails on their reverses")
a.set_ylabel("Pearson r")

fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(config.FIGURES + "/benchmark.png",bbox_inches="tight",facecolor="white")
print("saved benchmark.png")
