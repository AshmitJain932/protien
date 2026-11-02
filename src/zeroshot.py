#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""ESM2 zero-shot wt-marginal log-likelihood-ratio (LLR) feature.

For a mutation wt->mut at position i, using ONE forward pass of the base sequence:
    LLR = log P(mut | seq)_i  -  log P(wt | seq)_i
This is a strong, training-free predictor of mutation effect (ESM-1v style). It negates
under wt<->mut swap on the same base, so it is compatible with the anti-symmetric wrapper.
One forward pass per unique base sequence -> cheap even for a large model.
"""
import argparse, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, EsmForMaskedLM

def run(model_name, data, out, limit=None, batch=4):
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": batch = max(batch, 16)
    df = pd.read_csv(data)
    if limit: df = df.iloc[:limit].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device); model.eval()
    print(f"loaded {model_name} on {device}", flush=True)
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"}

    # unique base sequences -> positions needed
    from collections import defaultdict
    need = defaultdict(set)
    for r in df.itertuples():
        need[r.seq].add(r.pos0)
    seqs = list(need.keys())
    print(f"{model_name}: {len(df)} rows, {len(seqs)} unique base sequences", flush=True)

    logp = {}  # (seq,pos) -> np.array(logprob over 20 AAs)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(seqs), batch):
            chunk = seqs[b:b+batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            lp = torch.log_softmax(model(**enc).logits, dim=-1).float().cpu()  # [B,L+2,V]
            for j, s in enumerate(chunk):
                for pos in need[s]:
                    row = lp[j, pos+1]
                    logp[(s,pos)] = np.array([row[aa_ids[a]].item() for a in "ACDEFGHIKLMNPQRSTVWY"], dtype=np.float32)
            if b % (batch*20) == 0:
                el=time.time()-t0; done=b+len(chunk)
                print(f"  {done}/{len(seqs)}  ({el:.0f}s, {el/max(done,1)*1000:.0f} ms/seq)", flush=True)

    AAIDX = {a:k for k,a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    llr = np.zeros(len(df), dtype=np.float32)
    for k, r in enumerate(df.itertuples()):
        v = logp[(r.seq, r.pos0)]
        llr[k] = v[AAIDX[r.mut]] - v[AAIDX[r.wt]]
    np.savez(out, llr=llr, model=model_name)
    print(f"saved {out}  n={len(llr)}  total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--data", default=config.DATA + "/parsed/ddg_dataset.csv")
    ap.add_argument("--out", default=config.WORK + "/llr.npz")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.data, a.out, a.limit)
