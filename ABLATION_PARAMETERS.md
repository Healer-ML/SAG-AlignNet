## Ablation parameters

The following arguments are added for reviewer-requested ablations. Defaults keep
the original behavior as closely as possible.

| Design choice | Argument | Default | Meaning |
| --- | --- | --- | --- |
| Prototype number | `--num_prototypes` | `6` | Number of learnable prototypes used by PGU/prototype distillation. |
| Memory bank size | `--memory_size` | `-1` | Number of previous-epoch samples kept in the memory bank; `-1` keeps all samples. |
| Neighbor number | `--neighbor_k` | `100` | Top-k neighbors retrieved from the memory bank for soft target construction. |
| Soft-label mixing weight | `--soft_label_alpha` | `0.5` | Weight of soft ID loss in `(1-alpha) * hard_id + alpha * soft_id`. |

Example:

```bash
python train.py --num_prototypes 8 --memory_size 4096 --neighbor_k 50 --soft_label_alpha 0.3
```
