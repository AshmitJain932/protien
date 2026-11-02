#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Build the S669 comparison: our MEASURED numbers (computed in this repo on the exact verified
S669 split) alongside other tools' PUBLISHED numbers (clearly labelled, approximate). Emits a
markdown table and a bar figure that separates measured-here from literature and flags which
methods require a 3D structure."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def rj(p, default=None):
    try: return json.load(open(p))
    except Exception: return default

final = rj(config.RESULTS + "/results_final.json", {})
s669n = rj(config.RESULTS + "/results_s669.json", {})
unc = rj(config.RESULTS + "/results_uncertainty_650M.json") or rj(config.RESULTS + "/results_uncertainty_150M.json", {})
pro = rj(config.RESULTS + "/results_prospective.json", {})

struct = rj(config.RESULTS + "/results_struct.json", {})

def seq_r():
    if struct and "seq (bio+esm+llr)" in struct: return struct["seq (bio+esm+llr)"]["s669"]["pearson"]
    if final and "bio+esm+llr (FULL)" in final: return final["bio+esm+llr (FULL)"]["s669"]["pearson"]
    return float("nan")

our = seq_r()
our_struct = struct.get("+ structure", {}).get("s669", {}).get("pearson") if struct else None

# PUBLISHED S669 direct-Pearson values (approximate; from Pancotti et al. 2022 and follow-up papers).
# type: 'seq' = sequence-only, 'struct' = needs a 3D structure. Verify against the original papers.
LIT = [
    ("FoldX",            "struct", 0.22),
    ("Rosetta (cart)",   "struct", 0.39),
    ("ThermoNet",        "struct", 0.39),
    ("DDGun (seq)",      "seq",    0.41),
    ("INPS-seq",         "seq",    0.43),
    ("DDGun3D",          "struct", 0.43),
    ("ACDC-NN-seq",      "seq",    0.46),
    ("ACDC-NN",          "struct", 0.46),
    ("RaSP",             "struct", 0.46),
    ("PremPS",           "struct", 0.49),
    ("Stability Oracle", "struct", 0.53),
    ("DDMut",            "struct", 0.55),
    ("ThermoMPNN",       "struct", 0.58),
]

rows = []
if our_struct is not None:
    rows.append(("This work (structure-aware)", "struct", our_struct, "measured"))
rows.append(("This work (anti-sym, seq-only)", "seq", our, "measured"))
if s669n: rows.append(("  └ naive (no anti-sym)", "seq", s669n["naive"]["pearson"], "measured"))
if final and "zeroshot_LLR_only" in final:
    rows.append(("  └ ESM2 zero-shot only", "seq", final["zeroshot_LLR_only"]["s669"]["pearson"], "measured"))
rows += [(n, t, r, "literature") for n, t, r in LIT]

# --- markdown ---
md = ["# S669 blind-test comparison\n",
      "Pearson r on the S669 blind set (669 mutations, 94 proteins). "
      "'measured' rows are computed in this repo on our verified split; 'literature' rows are "
      "published values (approximate — verify against the original papers). 'struct' = requires a "
      "3D structure; 'seq' = sequence-only.\n",
      "| Method | Input | S669 Pearson r | Source |",
      "| --- | --- | --- | --- |"]
for n, t, r, src in sorted(rows, key=lambda x: (-(x[2] if x[2]==x[2] else -1))):
    md.append(f"| {n} | {t} | {r:.3f} | {src} |")
note = ""
if unc:
    k90 = unc.get("nominal_90", {})
    note = (f"\n**Beyond accuracy (measured here):** anti-symmetric by construction (reverse-law "
            f"error 0.0, symmetry bias ≈ 0). Calibrated 90% confidence intervals cover "
            f"{k90.get('s669_cov_norm', float('nan'))*100:.0f}% of blind S669 cases.")
if pro:
    note += (f" In-silico prospective design: predicted-stabilizing mutations are "
             f"{pro.get('enrichment_vs_base', float('nan')):.2f}x enriched for true stabilizers.")
md.append(note)
open(config.RESULTS + "/benchmark_table.md", "w").write("\n".join(md))
print("\n".join(md))

# --- figure ---
order = sorted(rows, key=lambda x: (x[2] if x[2]==x[2] else -1))
names = [n.strip() for n, _, _, _ in order]; vals = [r for _, _, r, _ in order]
colors = []
for n, t, r, src in order:
    if src == "measured" and n.startswith("This"): colors.append("#4f46e5")
    elif src == "measured": colors.append("#9aa3b5")
    elif t == "seq": colors.append("#0d8a89")
    else: colors.append("#c47d1c")
fig, ax = plt.subplots(figsize=(9, 6.5))
y = np.arange(len(names)); ax.barh(y, vals, color=colors)
for i, v in enumerate(vals): ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.set_xlabel("S669 Pearson r")
ax.set_xlim(0, 0.66); ax.grid(True, axis="x", color="#e6eaf1"); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4f46e5", label="this work (seq-only, measured)"),
                   Patch(color="#0d8a89", label="sequence-only (literature)"),
                   Patch(color="#c47d1c", label="structure-based (literature)")],
          frameon=False, fontsize=9, loc="lower right")
ax.set_title("ddG predictors on S669 (structure-based shown for context)", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(config.FIGURES + "/benchmark_table.png", bbox_inches="tight", facecolor="white")
print("\nsaved benchmark_table.md + benchmark_table.png")
