---
title: Denovo Enzyme Stability Workbench
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# Denovo — enzyme stability workbench

Predict how point mutations change an enzyme's folding stability **from sequence alone** (no 3D
structure), with a **calibrated trust badge** on every prediction, a **"what to test next"** list,
and a model that **learns your protein** from your own measured results.

Built on the anti-symmetric ΔΔG engine: https://github.com/AshmitJain932/protien

**Deploy:** create a Gradio Space on Hugging Face and upload the contents of this `app/` folder
(`app.py`, `features.py`, `ddg_dataset.csv`, `requirements.txt`, `README.md`). It runs on free CPU.
