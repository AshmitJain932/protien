#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Stage-2 figure: biophysical vs ESM2-augmented, both anti-symmetric."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

R=json.load(open(config.RESULTS + "/results_esm.json"))
d=np.load(config.WORK + "/preds_esm.npz")
INK="#1b2230"; MUT="#5b6475"; GRID="#e6eaf1"
C_CV="#4f46e5"; C_DIR="#0d8a89"; C_INV="#c47d1c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUT,
  "axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK,"figure.dpi":130})

fig,ax=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("ESM2-augmented anti-symmetric ΔΔG predictor (sequence-only, CPU)",
             fontsize=14,fontweight="bold",y=1.02)

# A: S2648 CV scatter (bio+esm)
a=ax[0]; y,p=d["y_cv"],d["p_cv"]; r=pearsonr(y,p)[0]; rho=spearmanr(y,p)[0]
rmse=np.sqrt(np.mean((y-p)**2))
a.scatter(y,p,s=10,alpha=0.45,color=C_CV,edgecolors="none")
a.plot([-7,7],[-7,7],"--",color=MUT,lw=1); a.set_xlim(-7,7); a.set_ylim(-7,7); a.set_aspect("equal")
a.grid(True,color=GRID,lw=0.8); a.set_axisbelow(True)
a.set_title(f"S2648 protein-blind 10-fold CV\nr={r:.3f}  ρ={rho:.3f}  RMSE={rmse:.2f}")
a.set_xlabel("experimental ΔΔG (kcal/mol)"); a.set_ylabel("predicted ΔΔG")

# B: Ssym dir+inv (bio+esm)
a=ax[1]
a.scatter(d["ydir"],d["pdir"],s=12,alpha=0.5,color=C_DIR,edgecolors="none",
          label=f"direct  r={pearsonr(d['ydir'],d['pdir'])[0]:.2f}")
a.scatter(d["yinv"],d["pinv"],s=12,alpha=0.5,color=C_INV,edgecolors="none",
          label=f"inverse r={pearsonr(d['yinv'],d['pinv'])[0]:.2f}")
a.plot([-7,7],[-7,7],"--",color=MUT,lw=1); a.set_xlim(-7,7); a.set_ylim(-7,7); a.set_aspect("equal")
a.grid(True,color=GRID,lw=0.8); a.set_axisbelow(True); a.legend(loc="upper left",frameon=False)
bias=R["bio+esm"]["ssym"]["bias"]
a.set_title(f"Ssym symmetry test\nbalanced · bias δ={bias:+.2f} kcal/mol")
a.set_xlabel("experimental ΔΔG"); a.set_ylabel("predicted ΔΔG")

# C: ablation bars (CV r and Ssym dir r across feature sets)
a=ax[2]
names=["bio","esm","bio+esm"]
cvr=[R[n]["cv"]["pearson"] for n in names]
ssr=[R[n]["ssym"]["dir"]["pearson"] for n in names]
x=np.arange(len(names)); w=0.36
b1=a.bar(x-w/2,cvr,w,color="#818cf8",label="S2648 CV  r")
b2=a.bar(x+w/2,ssr,w,color="#0d8a89",label="Ssym dir  r")
for b in list(b1)+list(b2):
    a.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.2f}",ha="center",fontsize=9)
a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0,0.8)
a.grid(True,axis="y",color=GRID,lw=0.8); a.set_axisbelow(True); a.legend(frameon=False,loc="upper left")
a.set_title("Feature ablation\n(ESM2 language-model signal lifts accuracy)")
a.set_ylabel("Pearson r")

fig.tight_layout()
fig.savefig(config.FIGURES + "/benchmark_esm.png",bbox_inches="tight",facecolor="white")
print("saved benchmark_esm.png")
