# Parameter definitions and sensitivity sweeps

`tools/run_sensitivity.py` runs one-factor-at-a-time training sweeps from each
AgriSearch dataset configuration. The figure parameters use the same symbols as
Figs. 8–9; each sweep changes one parameter while the other configuration values
stay fixed. Checkpoints are selected with validation mR.

## Parameters in Figs. 8–9

| Symbol | Config / CLI key | Code meaning | Values shown in the figures |
| --- | --- | --- | --- |
| α | `alpha` / `--alpha` | Weight of the current matching label in the neighbor-guided soft target (CProD). | `0.10, 0.25, 0.40, 0.55, 0.70, 0.85` |
| ρ | `rho` / `--rho` | Probability of selecting an eligible token for MLM masking (MasAlign). | `0.20, 0.25, 0.30, 0.35, 0.40, 0.55` |
| K | `K` / `--K` | Number of learnable prototypes. | `2, 4, 6, 8` |
| n | `n` / `--n` | Fraction of the per-rank training samples retained in the memory bank. | `0.25, 0.50, 0.75, 1.00` |
| m | `m` / `--m` | Number of nearest memory samples used to form neighbor targets. The effective value is capped by the memory-bank size. | `25, 50, 100, 200` |
| λ | `lambda_` / `--lambda` | Hard/soft matching-label loss mixing coefficient: `(1 − λ) × hard_matching_label + λ × soft_matching_label`. Epoch 1 still uses the hard matching-label loss. | `0, 0.25, 0.50, 0.75` |

The sensitivity runner's default grids for these six parameters match the values
above. To run a single sweep, for example:

```bash
python tools/run_sensitivity.py --dataset FRS-CMR --parameter alpha
python tools/run_sensitivity.py --dataset CCPRD --parameter lambda_ --dry-run
CUDA_VISIBLE_DEVICES=0,1 python tools/run_sensitivity.py \
  --dataset FRS-CMR --parameter m --seeds 1 2 3 --nproc-per-node 2
```

## Other implementation controls

The local prototype-alignment loss and matching-label loss are added to the
training objective with unit coefficients. They do not have separate loss-weight
settings.

| Key | Meaning | Runner default values |
| --- | --- | --- |
| `memory_size` | Optional absolute upper bound on memory samples; `-1` means no cap. | Three caps at 25%, 50%, and 75% of the configured capacity, plus the baseline. |
| `neighbor_weight` | Additive target weight for each retrieved neighbor. | `0, 0.00025, 0.0005, 0.001, 0.002` |
| `temperature` | Similarity-logit temperature (`logit_scale = 1 / temperature`). | `0.005, 0.01, 0.02, 0.05` |

Adam uses the separate keys `adam_beta1` and `adam_beta2`; they are not the
figure's α or λ. Legacy YAML names and CLI aliases (`soft_weight`, `mask_pro`,
`num_prototypes`, `memory_ratio`, `neighbor_k`, and `soft_label_alpha`) remain
supported. The former `id` loss-name token is normalized to `matching_label`.
The old Adam `alpha`/`beta` pair is supported when read from YAML.

Memory capacity is computed as `min(round(samples_per_rank × n), memory_size)`
when `memory_size > 0`; otherwise n controls capacity. The runner requires
`sampler: random` so its per-rank capacity calculation matches the training
loader, and rejects memory/neighbor sweeps that collapse to the same effective
setting.

The dataset configs contain the current values for the parameters listed above.
The GjPest4CMR config has α=`0.55`, while Fig. 8 marks `0.70` on the α curve.
Released checkpoints and their reported results remain from their original
runs; retraining uses the current config values and unit coefficients for the
two losses described above.

In the current forward pass, prototype neighbors are visited in a loop and the
soft matching-label targets are reassigned each iteration. The subsequent soft
matching-label loss therefore uses the last prototype's retrieved neighbors.
These sweeps reflect the implementation as it stands; aggregating neighbors
across prototypes would be a different experiment.
