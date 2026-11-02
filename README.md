# antisym-ddg — physically-consistent protein stability prediction

Predict the change in folding free energy (**ΔΔG**) when a protein is point-mutated — a core task
in enzyme and protein engineering (will this mutation make my enzyme more stable?).

This project is built around one principle most predictors ignore:

> ΔΔG of folding is a difference of free energies — a thermodynamic **state function** — so
> **ΔΔG(wt→mut) = −ΔΔG(mut→wt)** exactly. A correct predictor must be *anti-symmetric*.

Most ML ΔΔG tools are trained on datasets that are ~75% destabilizing mutations, so they quietly
learn "predict destabilizing": they score well on forward mutations but **fail on the reverse
mutations** and carry a large negative bias. Here anti-symmetry holds **by construction**, so the
reverse law is satisfied to machine precision and the model is honest in both directions — while
staying **sequence-only and CPU-only** (no 3D structure, no GPU).

## Results

All numbers are measured by the code in this repo. The headline is the **S669 blind test** (669
mutations whose proteins have **zero sequence overlap** with the training set).

| model (anti-symmetric) | S2648 protein-blind CV | S669 blind |
|---|---|---|
| biophysical features only | r = 0.351 | r = 0.328 |
| + ESM2 embedding deltas | r = 0.448 | r = 0.386 |
| **+ ESM2 zero-shot LLR (full)** | **r = 0.516** | **r = 0.413** |
| ESM2 zero-shot LLR alone (training-free) | r = 0.206 | r = 0.270 |

![final benchmark](figures/benchmark_final.png)

**What this does and does not beat — honestly:**
- **Raw accuracy:** S669 r ≈ 0.41 sits within the range of published *sequence-only* methods
  (~0.41–0.46, e.g. DDGun, ACDC-NN-seq) and **below** the best structure- or large-PLM-based tools
  (~0.5–0.58). This project does **not** claim a state-of-the-art accuracy win.
- **Physical consistency:** here it wins outright. The forward/reverse law holds exactly and the
  symmetry bias is ≈0, where the standard approach collapses.

### The anti-symmetry win (clean head-to-head, biophysical features, Ssym)

| model | direct r | inverse r | exact reverse-law error |
|---|---|---|---|
| naive (standard ML) | 0.715 | **0.043** | 5.75 kcal/mol |
| **anti-symmetric (ours)** | 0.576 | **0.579** | **0.000000** |

![anti-symmetry](figures/benchmark.png)

The naive model looks best on direct mutations but cannot predict their reverses; ours is balanced
and obeys the law exactly. (Ssym shares 12/14 proteins with the training set — standard in the
literature but it inflates Ssym for every method trained on S2648, so **S669 is the number to
trust**.)

## Why it is anti-symmetric by construction

Every input feature is a sign-flipping quantity, so the whole feature vector `F` satisfies
`F(mut→wt) = −F(wt→mut)`:

- residue **property deltas** `p(mut) − p(wt)` — hydropathy, volume, charge, polarity, helix &
  sheet propensity, flexibility, H-bonding, aromaticity, mass;
- **delta × environment** interactions, where the environment (local burial, local charge,
  distance from termini) is computed from the *neighbours* and so is swap-invariant;
- an **ESM2 embedding delta** `ESM(seq_mut)[i] − ESM(seq_wt)[i]` (same base sequence);
- an **ESM2 zero-shot LLR** `log P(mut) − log P(wt)` at the mutated position (one forward pass).

Wrapping *any* learner `h` as `ddG = (h(F) − h(−F)) / 2` is then exactly anti-symmetric regardless
of `h`. The learner here is gradient-boosted trees (XGBoost), trained on the data augmented with
every reverse mutation. See [`docs/METHOD.md`](docs/METHOD.md).

## Data

| set | role | mutations | proteins | source |
|---|---|---|---|---|
| **S2648** | training / blind CV | 2,579 | 127 | ACDC-NN repo |
| **Ssym** | symmetry test | 341 dir + 341 inv | 15 | ACDC-NN repo |
| **S669** | blind generalization test | 669 | 94 | ThermoMPNN repo |

Benchmark files give only `PDBchain mutation ΔΔG` with **no sequences**. Sequences and the mutated
position are reconstructed directly from RCSB PDB structures and **every mutation's wild-type
residue is verified against the structure** (match rate 97–100%). S2648 and Ssym use 1-based
sequential numbering over observed residues; S669 uses PDB author numbering — both are handled and
validated. The committed `data/parsed/*.csv` already contain the verified, sequence-bearing data.

## Repository layout

```
src/            all pipeline code (portable; paths via src/config.py, override with PROTIEN_ROOT)
  config.py         central paths
  features.py       anti-symmetric biophysical features (+ exact-symmetry self-test)
  esm_features.py   ESM2 embedding-delta extractor
  zeroshot.py       ESM2 zero-shot LLR feature
  build_dataset.py  build + verify S2648 / Ssym from PDBs
  build_s669.py     build + verify S669 (author numbering) + overlap check
  train.py          biophysical model: CV, Ssym, naive ablation, exact-law proof
  train_esm.py      bio + ESM2 ablation
  train_full.py     final bio + ESM2 + LLR model
  eval_s669.py      blind S669 evaluation
  fit_predictor.py  fit & save the deployable biophysical model
  predict.py        CLI: score mutations, shows the exact reverse
  make_report*.py   benchmark figures
scripts/get_data.sh clone sources + download PDBs + build the parsed datasets
data/parsed/        verified datasets (committed); raw clones + PDBs are gitignored
models/             trained models (biophysical CLI model + full model)
figures/ results/   reference outputs
docs/METHOD.md      the method in detail
```

## Install

```bash
pip install -r requirements.txt
```

## Quickstart — score mutations (no GPU, no download)

```bash
cd src
python predict.py <SEQUENCE> M1A V47I ...
# e.g. hen egg white lysozyme:
python predict.py KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL W62G D18A
```

## Reproduce the benchmarks

```bash
bash scripts/get_data.sh          # clone sources, download PDBs, build+verify datasets (network)
cd src
python train.py                   # biophysical model + anti-symmetry proof
python esm_features.py            # ESM2 embeddings  (CPU, a few minutes; downloads the model)
python esm_features.py --data ../data/parsed/s669_dataset.csv --out ../artifacts/esm_delta_s669.npz
python zeroshot.py                # ESM2 zero-shot LLR (train set)
python zeroshot.py --data ../data/parsed/s669_dataset.csv --out ../artifacts/llr_s669.npz
python train_full.py              # final fused model: S2648 CV + S669 blind
python make_report_final.py       # figure
```

## Honest limitations

- **Sequence-only**: no explicit 3D contacts; "burial" is a hydropathy proxy. Structure features
  would raise accuracy (and could enter the same anti-symmetric recipe as swap-invariant terms).
- **Small, noisy labels**: experimental ΔΔG error alone is ~0.5 kcal/mol and the data is biased
  toward a few well-studied proteins, so protein-blind correlation is modest for everyone here.
- **ESM2-35M embeddings / 150M zero-shot** were used to fit the CPU budget; larger PLMs should help.
- Predicts folding **stability**, not activity / expression / solubility.

## Roadmap

1. Larger PLM (ESM2-650M / 3B) for embeddings and zero-shot — the biggest accuracy lever.
2. Structure-aware swap-invariant environment features (SASA, contact number) from predicted or
   experimental structures.
3. Train on the Tsuboyama/Rocklin mega-scale stability dataset (~800k variants) — needs a GPU.
4. Extend the anti-symmetric recipe to solubility and expression prediction.

## Data sources & references (verify against originals)

- S2648 / Ssym and the CV splits: ACDC-NN — Benevenuta, Birolo et al.; `github.com/compbiomed-unito/acdc-nn`.
- S669 blind set (via ThermoMPNN): Pancotti et al., *Brief. Bioinform.* 2022; Dieckhaus et al. (ThermoMPNN) 2024; `github.com/Kuhlman-Lab/ThermoMPNN`.
- Ssym symmetry benchmark: Pucci, Usmanova et al. 2018.
- ESM2 protein language model: Lin et al., *Science* 2023.
- Biophysical scales: Kyte–Doolittle; Pace & Scholtz (helix); Chou–Fasman (sheet); Vihinen (flexibility).

## License

MIT — see [LICENSE](LICENSE).
