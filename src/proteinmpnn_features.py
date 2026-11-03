#!/usr/bin/env python3
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""ProteinMPNN inverse-folding feature (structure-conditioned, anti-symmetric).

Uses ProteinMPNN (Dauparas et al., Baker lab; https://github.com/dauparas/ProteinMPNN, MIT) as a
frozen feature extractor. For a mutation at position i it computes
    LLR = log P(mut | backbone)_i  -  log P(wt | backbone)_i
from ProteinMPNN's per-position amino-acid distribution (averaged over K random decoding orders).
This is one of the strongest structure-based stability signals and negates under wt<->mut swap, so
it is compatible with the anti-symmetric wrapper. ProteinMPNN is third-party; the integration here
is ours.
"""
import argparse, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")

MPNN_DIR = "/home/user/data_src/ProteinMPNN"
_sys.path.insert(0, MPNN_DIR)
from protein_mpnn_utils import parse_PDB, tied_featurize, ProteinMPNN

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AAI = {a: i for i, a in enumerate(ALPHABET)}

def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                    num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
                    k_neighbors=ckpt["num_edges"])
    m.load_state_dict(ckpt["model_state_dict"]); m.to(device); m.eval()
    return m

_cache = {}
def chain_logp(model, pdb, chain, pdbdir, device, K=8):
    """Average per-position log-prob [L,21] and native sequence for one chain."""
    key = (pdb, chain)
    if key in _cache: return _cache[key]
    try:
        pdb_list = parse_PDB(f"{pdbdir}/{pdb.upper()}.pdb", input_chain_list=[chain])
    except Exception:
        _cache[key] = (None, None); return None, None
    if not pdb_list:
        _cache[key] = (None, None); return None, None
    batch = [pdb_list[0]]
    feat = tied_featurize(batch, device, None, None, None, None, None, None, ca_only=False)
    X, S, mask, _, chain_M, chain_encoding_all = feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]
    residue_idx, chain_M_pos = feat[12], feat[10]
    acc = torch.zeros(S.shape[1], 21)
    with torch.no_grad():
        for _ in range(K):
            randn = torch.randn(chain_M.shape, device=device)
            lp = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn)
            acc += lp[0].cpu()
    acc /= K
    native = "".join(ALPHABET[i] for i in S[0].cpu().numpy())
    _cache[key] = (acc.numpy(), native)
    return _cache[key]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA + "/parsed/ddg_dataset.csv")
    ap.add_argument("--out", default=config.WORK + "/mpnn_llr.npz")
    ap.add_argument("--pdbdir", default="/home/user/data_src/pdbs")
    ap.add_argument("--weights", default=MPNN_DIR + "/vanilla_model_weights/v_48_020.pt")
    ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    model = load_model(a.weights, device)
    print(f"ProteinMPNN loaded on {device}", flush=True)

    df = pd.read_csv(a.data).reset_index(drop=True)
    llr = np.zeros(len(df), dtype=np.float32); ok = 0
    for k, r in enumerate(df.itertuples()):
        lp, native = chain_logp(model, r.pdb, r.chain, a.pdbdir, device, a.K)
        if lp is None: continue
        i = int(r.pos0)
        if i >= len(native) or native[i] != r.wt: continue
        llr[k] = lp[i, AAI[r.mut]] - lp[i, AAI[r.wt]]; ok += 1
        if k % 500 == 0: print(f"  {k}/{len(df)}", flush=True)
    np.savez(a.out, llr=llr)
    print(f"ProteinMPNN LLR: {ok}/{len(df)} rows mapped -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
