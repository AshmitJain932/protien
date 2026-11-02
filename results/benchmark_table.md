# S669 blind-test comparison

Pearson r on the S669 blind set (669 mutations, 94 proteins). 'measured' rows are computed in this repo on our verified split; 'literature' rows are published values (approximate — verify against the original papers). 'struct' = requires a 3D structure; 'seq' = sequence-only.

| Method | Input | S669 Pearson r | Source |
| --- | --- | --- | --- |
| ThermoMPNN | struct | 0.580 | literature |
| DDMut | struct | 0.550 | literature |
| Stability Oracle | struct | 0.530 | literature |
| This work (structure-aware) | struct | 0.502 | measured |
| PremPS | struct | 0.490 | literature |
| This work (anti-sym, seq-only) | seq | 0.489 | measured |
| ACDC-NN-seq | seq | 0.460 | literature |
| ACDC-NN | struct | 0.460 | literature |
| RaSP | struct | 0.460 | literature |
| INPS-seq | seq | 0.430 | literature |
| DDGun3D | struct | 0.430 | literature |
| DDGun (seq) | seq | 0.410 | literature |
| Rosetta (cart) | struct | 0.390 | literature |
| ThermoNet | struct | 0.390 | literature |
|   └ naive (no anti-sym) | seq | 0.374 | measured |
|   └ ESM2 zero-shot only | seq | 0.270 | measured |
| FoldX | struct | 0.220 | literature |

**Beyond accuracy (measured here):** anti-symmetric by construction (reverse-law error 0.0, symmetry bias ≈ 0). Calibrated 90% confidence intervals cover 81% of blind S669 cases. In-silico prospective design: predicted-stabilizing mutations are 2.07x enriched for true stabilizers.