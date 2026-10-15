## Hyperparameters

Dataset-specific values in `configs/` follow the per-dataset sensitivity
results in Figures 8 and 9 of the paper. Shared defaults are listed below.

| Design choice | Argument | Default | Meaning |
| --- | --- | --- | --- |
| Local prototype alignment weight | `--cprod_alpha` | `1.0` | Weight applied to the prototype-level alignment loss. |
| Prototype number | `--num_prototypes` | `6` | Number of learnable prototypes used by PGU/prototype distillation. |
| Memory bank size | `--memory_size` | `-1` | Number of previous-epoch samples kept in the memory bank; `-1` keeps all samples. |
| Memory bank ratio | `--memory_ratio` | `1.0` | Fraction of current training pairs retained in the memory bank. |
| Neighbor number | `--neighbor_k` | `100` | Top-k neighbors retrieved from the memory bank for soft target construction. |
| Soft-label mixing weight | `--soft_label_alpha` | `0.5` | Weight of soft ID loss in `(1-lambda) * hard_id + lambda * soft_id`. |
| Gradient clipping | `--grad_clip_max_norm` | `5.0` | Maximum gradient norm; values at or below zero disable clipping. |

Example:

```bash
python train.py --num_prototypes 8 --memory_size 4096 --neighbor_k 50 --soft_label_alpha 0.3
```
