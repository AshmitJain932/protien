#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""ESM2 embedding-delta features (anti-symmetry preserving).

For a mutation at position i, the feature block is
    emb_delta = ESM(seq_with_mut)[i] - ESM(seq_with_wt)[i]
computed from the SAME base sequence, so swapping wt<->mut negates it exactly.
We extract only the needed positions per unique sequence to keep memory bounded.
"""
import sys, time, numpy as np, pandas as pd, torch
from collections import defaultdict
from transformers import AutoTokenizer, EsmModel

MODEL = "facebook/esm2_t12_35M_UR50D"   # 480-dim, CPU-friendly
DATA = config.DATA + "/parsed/ddg_dataset.csv"
OUT = config.WORK + "/esm_delta.npz"

def main(limit=None, batch=8, data=DATA, out=OUT, model_name=MODEL):
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": batch = max(batch, 32)   # GPUs like bigger batches
    df = pd.read_csv(data)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    # collect needed (sequence, position) extractions
    need = defaultdict(set)   # seq -> {positions}
    reqs = []                 # per row: (base_seq, mut_seq, pos)
    for r in df.itertuples():
        base = r.seq
        mut = base[:r.pos0] + r.mut + base[r.pos0 + 1:]
        need[base].add(r.pos0); need[mut].add(r.pos0)
        reqs.append((base, mut, r.pos0))
    seqs = list(need.keys())
    print(f"rows={len(df)}  unique sequences to embed={len(seqs)}", flush=True)

    print(f"loading {model_name} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device); model.eval()
    D = model.config.hidden_size

    vec = {}  # (seq, pos) -> np.array(D)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(seqs), batch):
            chunk = seqs[b:b + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            hs = model(**enc).last_hidden_state.float().cpu()   # [B, L+2, D], token 0 = CLS
            for j, s in enumerate(chunk):
                for pos in need[s]:
                    vec[(s, pos)] = hs[j, pos + 1].numpy().astype(np.float32)
            if b % (batch * 25) == 0:
                el = time.time() - t0
                done = b + len(chunk)
                print(f"  embedded {done}/{len(seqs)}  ({el:.0f}s, {el/max(done,1)*1000:.0f} ms/seq)", flush=True)

    delta = np.zeros((len(df), D), dtype=np.float32)
    for k, (base, mut, pos) in enumerate(reqs):
        delta[k] = vec[(mut, pos)] - vec[(base, pos)]
    np.savez(out, delta=delta, dim=D)
    print(f"saved {out}  shape={delta.shape}  total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    main(limit=a.limit, data=a.data, out=a.out, model_name=a.model)
