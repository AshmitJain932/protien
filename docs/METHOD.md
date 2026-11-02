# Method

## The problem

Given a protein sequence and a point mutation (wild-type residue → mutant at position *i*),
predict **ΔΔG**, the change in folding free energy. Negative = destabilizing, positive =
stabilizing (the convention of the S2648 / Ssym / S669 labels used here).

## The physical constraint

Folding ΔΔG is the difference of two folding free energies:

```
ΔΔG(wt→mut) = ΔG_fold(mut) − ΔG_fold(wt)
```

Swapping the roles of wild-type and mutant therefore negates it exactly:

```
ΔΔG(mut→wt) = −ΔΔG(wt→mut)        (anti-symmetry)
```

This is not a modelling choice; it is a property of a state-function difference. Yet most ML
predictors violate it, because they are trained on datasets dominated by destabilizing mutations
and learn a destabilization prior. On the Ssym benchmark this shows up as a large negative bias and
a collapse of the correlation on the inverse mutations.

## Anti-symmetry by construction

We make the predictor anti-symmetric **structurally**, so it cannot violate the law regardless of
how the learner is fit.

### 1. Sign-flipping features

Every feature negates under the wt↔mut swap, so the whole feature vector obeys
`F(mut→wt) = −F(wt→mut)` (verified exactly in `features.py`'s self-test):

- **Property deltas** `p(mut) − p(wt)` for 11 residue scales: Kyte–Doolittle hydropathy, residue
  volume, charge, Grantham polarity, Pace–Scholtz helix propensity, Chou–Fasman sheet propensity,
  Vihinen flexibility, H-bond donors, H-bond acceptors, aromaticity, mass.
- **Delta × environment** interactions. The environment descriptors — local burial (mean hydropathy
  of neighbours), local charge density, normalized distance from the termini — are computed from the
  residues *around* position *i*, **excluding** *i* itself, so they are invariant under the swap.
  A (sign-flipping delta) × (invariant environment) term still flips sign. This lets context modulate
  the effect without breaking anti-symmetry.
- **ESM2 embedding delta** `ESM(seq_mut)[i] − ESM(seq_wt)[i]`, both embeddings taken from the same
  base sequence so the difference flips sign under the swap.
- **ESM2 zero-shot LLR** `log P(mut | seq)_i − log P(wt | seq)_i` from one forward pass of the base
  sequence (ESM-1v-style wt-marginal scoring); flips sign under the swap.

### 2. Anti-symmetric wrapper

For any learner `h`, define

```
ddG_pred(F) = ( h(F) − h(−F) ) / 2
```

This function is odd by construction, so `ddG_pred(−F) = −ddG_pred(F)`. Because the real reverse
mutation has feature vector `−F`, the predicted reverse equals the negated forward prediction
**exactly** — machine-precision 0.0 in the repo's check, versus ~5.75 kcal/mol violation for the
same learner used naively.

### 3. Learner and training

`h` is a gradient-boosted tree ensemble (XGBoost, depth 4, 600–700 trees). Training data is
**augmented** with every reverse mutation (`F → −F`, `y → −y`), which also balances the label
distribution and removes the destabilization prior.

## Evaluation protocol

- **S2648**: 10-fold cross-validation using the predefined, protein-blind splits shipped with
  ACDC-NN (verified here to have zero cross-fold protein overlap). Metrics are computed on the
  direct mutations with the anti-symmetric wrapper.
- **Ssym**: train on all of S2648, test on the direct and inverse sets; report both correlations
  and the bias `δ = mean(pred_dir) + mean(pred_inv)`. Note Ssym shares 12/14 proteins with S2648.
- **S669**: the honest blind test — 94 proteins with zero sequence overlap with S2648 (checked by
  exact match and 6-mer Jaccard). Train on all S2648, predict S669.

## Sequence reconstruction and verification

Benchmark files provide only `PDBchain mutation ΔΔG`. For each entry we parse the RCSB structure
(first model, Cα atoms, altloc blank/A, common modified residues mapped), reconstruct the observed
sequence, map the mutation position, and **verify the wild-type residue matches**. S2648/Ssym use
1-based sequential numbering over observed residues; S669 uses PDB author numbering. Entries whose
wild-type residue cannot be verified are dropped (2.6% of S2648; 0% of S669).
