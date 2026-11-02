#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Final figure: full-model blind scatter + the feature-ablation climb."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
R=json.load(open(config.RESULTS + "/results_final.json")); d=np.load(config.WORK + "/preds_full.npz")
INK="#1b2230"; MUT="#5b6475"; GRID="#e6eaf1"; C="#4f46e5"; C2="#0d8a89"; C3="#c47d1c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUT,
  "axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK,"figure.dpi":130})
fig,ax=plt.subplots(1,2,figsize=(11,5.2))
fig.suptitle("Final anti-symmetric model — biophysics + ESM2 embeddings + zero-shot LLR (sequence-only, CPU)",
             fontsize=12.5,fontweight="bold",y=1.0)
a=ax[0]; y,p=d["y669"],d["p669"]
a.scatter(y,p,s=12,alpha=0.5,color=C,edgecolors="none")
a.plot([-7,7],[-7,7],"--",color=MUT,lw=1); a.set_xlim(-7,7);a.set_ylim(-7,7);a.set_aspect("equal")
a.grid(True,color=GRID,lw=0.8); a.set_axisbelow(True)
s=R["bio+esm+llr (FULL)"]["s669"]
a.set_title(f"S669 blind test (669 muts, 0 overlap)\nr={s['pearson']:.3f}  ρ={s['spearman']:.3f}  RMSE={s['rmse']:.2f}")
a.set_xlabel("experimental ΔΔG (kcal/mol)"); a.set_ylabel("predicted ΔΔG")
a=ax[1]
names=["bio","bio+esm","FULL"]; keys=["bio","bio+esm","bio+esm+llr (FULL)"]
cv=[R[k]["cv"]["pearson"] for k in keys]; bl=[R[k]["s669"]["pearson"] for k in keys]
x=np.arange(3); w=0.36
b1=a.bar(x-w/2,cv,w,color=C,label="S2648 blind CV"); b2=a.bar(x+w/2,bl,w,color=C2,label="S669 blind")
for b in list(b1)+list(b2): a.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.2f}",ha="center",fontsize=9)
a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0,0.62)
a.grid(True,axis="y",color=GRID,lw=0.8); a.set_axisbelow(True); a.legend(frameon=False,loc="upper left")
a.set_title("each signal adds real accuracy\n(anti-symmetry preserved throughout)")
a.set_ylabel("Pearson r")
fig.tight_layout(); fig.savefig(config.FIGURES + "/benchmark_final.png",bbox_inches="tight",facecolor="white")
print("saved benchmark_final.png")
