#!/usr/bin/env python3
"""Denovo — enzyme stability workbench (web app on top of the anti-symmetric ddG engine).

What makes it different from other ddG tools:
  • a calibrated TRUST badge + confidence interval on every prediction (not a bare number),
  • a "what to test next" list (experiment design, by uncertainty),
  • "teach it": paste your measured results and the model re-fits to YOUR protein.
Runs the real anti-symmetric model (biophysical, CPU, no structure needed).
"""
import os, sys, numpy as np, pandas as pd, gradio as gr
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # self-contained (features.py beside app.py)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))       # or run from the repo
import features as F
from xgboost import XGBRegressor

# data: local copy (Hugging Face Space) or the repo's parsed file
DDG = (os.path.join(_HERE, "ddg_dataset.csv") if os.path.exists(os.path.join(_HERE, "ddg_dataset.csv"))
       else os.path.join(_HERE, "..", "data", "parsed", "ddg_dataset.csv"))
AAs = list("ACDEFGHIKLMNPQRSTVWY")

def anti(m, X): return (m.predict(X) - m.predict(-X)) / 2.0
def augment(X, y): return np.vstack([X, -X]), np.concatenate([y, -y])
def mk(seed): return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=1.5, min_child_weight=3, n_jobs=4, random_state=seed)
def fit_ensemble(X, y, K=5, w=None):
    Xa, ya = augment(X, y)
    wa = np.concatenate([w, w]) if w is not None else None
    return [mk(100 + 11 * k).fit(Xa, ya, sample_weight=wa) for k in range(K)]
def ens(models, X):
    P = np.stack([anti(m, X) for m in models], 0)
    return P.mean(0), P.std(0)

# ---- startup: train engine + calibrate uncertainty on S2648 ----
print("loading engine ...", flush=True)
_df = pd.read_csv(DDG); _s = _df[_df.dataset == "S2648"].reset_index(drop=True)
BASE_X = F.featurize_df(_s).astype(np.float32); BASE_Y = _s.ddg.values; FOLDS = _s.fold.values
_cal_models = fit_ensemble(BASE_X[FOLDS != 9], BASE_Y[FOLDS != 9])
_pc, _sc = ens(_cal_models, BASE_X[FOLDS == 9])
_resid = np.abs(BASE_Y[FOLDS == 9] - _pc)
Q90 = float(np.quantile(_resid, 0.9)); Q80 = float(np.quantile(_resid, 0.8))
STD_MED = float(np.median(_sc)); STD_HI = float(np.quantile(_sc, 0.80))
SERVING = fit_ensemble(BASE_X, BASE_Y)
print(f"engine ready. conformal 90% half-width = {Q90:.2f} kcal/mol", flush=True)

_VALID = set("ACDEFGHIKLMNPQRSTVWY")
def clean(seq):
    return "".join(c for c in seq.upper() if c in _VALID)

def trust_badge(std):
    if std <= STD_MED: return "🟢 High"
    if std <= STD_HI:  return "🟡 Medium"
    return "⚪ Low (uncertain)"

def scan(seq, models):
    seq = clean(seq)
    rows, feats = [], []
    for i in range(len(seq)):
        wt = seq[i]
        for mut in AAs:
            if mut == wt: continue
            rows.append((i, wt, mut)); feats.append(F.featurize(seq, i, wt, mut))
    X = np.asarray(feats, dtype=np.float32)
    mean, std = ens(models, X)
    df = pd.DataFrame({
        "mutation": [f"{wt}{i+1}{mut}" for (i, wt, mut) in rows],
        "pred_ddG": mean, "std": std,
        "lo90": mean - Q90, "hi90": mean + Q90,
    })
    return seq, df

def fmt_pred(df, by="stab", n=15):
    d = df.copy()
    d = d.sort_values("pred_ddG", ascending=(by != "stab"))
    if by == "test":  # most informative = highest model uncertainty
        d = df.sort_values("std", ascending=False)
    d = d.head(n)
    out = pd.DataFrame({
        "Mutation": d.mutation,
        "ΔΔG (kcal/mol)": d.pred_ddG.map(lambda v: f"{v:+.2f}"),
        "90% interval": [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(d.lo90, d.hi90)],
        "Effect": d.pred_ddG.map(lambda v: "stabilizing" if v > 0.3 else ("destabilizing" if v < -0.3 else "~neutral")),
        "Trust": d["std"].map(trust_badge),
    })
    return out

def analyze(seq, state):
    seq = clean(seq)
    if len(seq) < 10:
        return "Paste a protein sequence (≥10 residues).", None, None, state
    models = state or SERVING
    seq, df = scan(seq, models)
    summ = (f"**{len(seq)} residues** · scanned **{len(df)}** possible point mutations · "
            f"model: anti-symmetric ensemble (sequence-only, no structure needed).\n\n"
            f"Sign: **positive ΔΔG = stabilizing**, negative = destabilizing. "
            f"Trust badge reflects model agreement; 90% interval is conformal-calibrated "
            f"(±{Q90:.1f} kcal/mol).")
    state = {"seq": seq, "df": df, "models": models}
    return summ, fmt_pred(df, "stab"), fmt_pred(df, "test"), state

def teach(results_text, state):
    if not state or "seq" not in state:
        return "Run **Analyze** first, then teach the model.", None, None, state
    seq = state["seq"]
    X_add, y_add = [], []
    parsed = 0
    for line in results_text.replace(",", " ").splitlines():
        p = line.split()
        if len(p) < 2: continue
        mut, val = p[0].strip().upper(), p[1]
        try:
            wt, mt, pos = mut[0], mut[-1], int(mut[1:-1]); val = float(val)
        except (ValueError, IndexError):
            continue
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt: continue
        X_add.append(F.featurize(seq, pos - 1, wt, mt)); y_add.append(val); parsed += 1
    if parsed == 0:
        return ("No valid rows. Use lines like `W62G -1.3` (mutation then measured ΔΔG; "
                "wild-type letter must match this sequence)."), None, None, state
    # refit: base data + user points upweighted so the model adapts to THIS protein
    Xn = np.vstack([BASE_X] + X_add * 20)
    yn = np.concatenate([BASE_Y, np.repeat(y_add, 20)])
    w = np.concatenate([np.ones(len(BASE_X)), np.full(parsed * 20, 1.0)])
    models = fit_ensemble(Xn, yn, w=w)
    _, df = scan(seq, models)
    state = {"seq": seq, "df": df, "models": models}
    msg = (f"✅ Model updated with **{parsed}** measured result(s) for this protein. "
           f"Predictions and the test-next list below are now personalized — re-fit live.")
    return msg, fmt_pred(df, "stab"), fmt_pred(df, "test"), state

EXAMPLE = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"

with gr.Blocks(title="Denovo — enzyme stability workbench") as demo:
    gr.Markdown("# 🧬 Denovo — enzyme stability workbench\n"
                "Predict how mutations change an enzyme's folding stability — **from sequence alone, "
                "no 3D structure needed** — with a **calibrated trust badge** on every prediction, a "
                "**ranked list of what to test next**, and a model that **learns your protein** from "
                "your own results. Built on an anti-symmetric engine that obeys the thermodynamic "
                "reverse-mutation law exactly.")
    st = gr.State()
    seq_in = gr.Textbox(label="Protein sequence (or FASTA)", value=EXAMPLE, lines=3)
    with gr.Row():
        go = gr.Button("Analyze", variant="primary")
        ex = gr.Button("Load example (lysozyme)")
    summary = gr.Markdown()
    with gr.Tab("Top predicted-stabilizing"):
        t_stab = gr.Dataframe(label="Ranked by predicted stabilization (positive = more stable)")
    with gr.Tab("🧪 What to test next"):
        gr.Markdown("The mutations where the model is **least certain** — testing these teaches it the most per experiment.")
        t_test = gr.Dataframe()
    with gr.Tab("🎓 Teach it (learn your protein)"):
        gr.Markdown("Paste measured results, one per line: `mutation  measuredΔΔG` (e.g. `W62G -1.3`). "
                    "The model re-fits to this protein, live.")
        res_in = gr.Textbox(label="Your measured results", lines=5, placeholder="W62G -1.3\nD18A  0.4\nI55V  0.1")
        learn = gr.Button("Update model with my results", variant="primary")
        learn_msg = gr.Markdown()
    go.click(analyze, [seq_in, st], [summary, t_stab, t_test, st])
    ex.click(lambda: EXAMPLE, None, seq_in)
    learn.click(teach, [res_in, st], [learn_msg, t_stab, t_test, st])
    demo.load(analyze, [seq_in, st], [summary, t_stab, t_test, st])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                share=os.environ.get("SHARE", "0") == "1", theme=gr.themes.Soft())
