#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""S669 blind-test figure: anti-symmetric vs naive, accuracy and bias."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

d=np.load(config.WORK + "/preds_s669.npz"); R=json.load(open(config.RESULTS + "/results_s669.json"))
INK="#1b2230"; MUT="#5b6475"; GRID="#e6eaf1"; C_ANTI="#4f46e5"; C_NAIVE="#d13b3b"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUT,
  "axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK,"figure.dpi":130})
y,p,pn=d["y"],d["pred"],d["predn"]

fig,ax=plt.subplots(1,2,figsize=(10.6,5.2))
fig.suptitle("S669 blind test — 669 mutations, 94 proteins, zero overlap with training",
             fontsize=13,fontweight="bold",y=1.0)
a=ax[0]
a.scatter(y,p,s=12,alpha=0.5,color=C_ANTI,edgecolors="none")
a.plot([-7,7],[-7,7],"--",color=MUT,lw=1); a.set_xlim(-7,7); a.set_ylim(-7,7); a.set_aspect("equal")
a.grid(True,color=GRID,lw=0.8); a.set_axisbelow(True)
a.set_title("anti-symmetric bio+ESM2 (sequence-only)\nr={:.3f}  ρ={:.3f}  RMSE={:.2f}  bias={:+.2f}".format(
    R['anti']['pearson'],R['anti']['spearman'],R['anti']['rmse'],R['anti']['mean_pred']))
a.set_xlabel("experimental ΔΔG (kcal/mol)"); a.set_ylabel("predicted ΔΔG")

a=ax[1]
labels=["Pearson r","mean pred\n(bias→0 ideal)","exp mean"]
anti=[R['anti']['pearson'],R['anti']['mean_pred'],R['exp_mean']]
naive=[R['naive']['pearson'],R['naive']['mean_pred'],R['exp_mean']]
x=np.arange(3); w=0.36
b1=a.bar(x-w/2,naive,w,color=C_NAIVE,label="naive")
b2=a.bar(x+w/2,anti,w,color=C_ANTI,label="anti-symmetric (ours)")
for b in list(b1)+list(b2):
    a.text(b.get_x()+b.get_width()/2, b.get_height()+(0.02 if b.get_height()>=0 else -0.08),
           f"{b.get_height():.2f}",ha="center",fontsize=9)
a.axhline(0,color=MUT,lw=0.8); a.set_xticks(x); a.set_xticklabels(labels)
a.grid(True,axis="y",color=GRID,lw=0.8); a.set_axisbelow(True); a.legend(frameon=False)
a.set_title("accuracy similar, but the naive model carries a\nlarge destabilization bias; ours is ~unbiased")
fig.tight_layout()
fig.savefig(config.FIGURES + "/benchmark_s669.png",bbox_inches="tight",facecolor="white")
print("saved benchmark_s669.png")
