# Code-level sensitivity analysis

`tools/run_sensitivity.py` runs fresh, one-factor-at-a-time training jobs from a
dataset YAML. The selected argument changes between runs; the other dataset
settings stay fixed. Every run selects its checkpoint with validation mR and
records the resolved command, configuration, seed, best epoch, validation mR,
effective memory size, and effective neighbor count in JSON and CSV files.

## Active arguments

| Argument | What the implementation changes | Default sweep values |
| --- | --- | --- |
| `cprod_alpha` | Multiplier on the local prototype-alignment loss; `id_loss_weight` remains fixed. | `0, 0.25, 0.5, 0.75, 1` |
| `mask_pro` | Selection probability for eligible MLM tokens. The current replacement thresholds turn about 99.99% of selected tokens into mask tokens. | `0.1, 0.2, 0.3, 0.4, 0.5, 0.6` |
| `num_prototypes` | Number of learnable prototypes and prototype-index classes. | `2, 4, 6, 8` |
| `memory_ratio` | Fraction of training samples retained in the epoch memory bank. | `0.25, 0.5, 0.75, 1` |
| `memory_size` | Absolute upper bound on memory samples; `-1` means no cap. | Three caps at 25%, 50%, and 75% of the configured effective capacity, plus the configured baseline. |
| `neighbor_k` | Requested number of nearest memory samples used to form neighbor targets. | `25, 50, 100, 200` |
| `neighbor_weight` | Additive target weight for each retrieved neighbor. Its default `0.001` preserves the previous hard-coded behavior. | `0, 0.00025, 0.0005, 0.001, 0.002` |
| `soft_label_alpha` | Soft-ID loss mixing weight in `(1 - value) * hard_ID + value * soft_ID`; the first epoch still uses hard ID loss. | `0, 0.25, 0.5, 0.75, 1` |
| `soft_weight` | Raw target weight assigned to the current identity when constructing soft ID targets. | `0, 0.25, 0.5, 0.75, 1` |
| `temperature` | Temperature used to scale the image-text and prototype similarity logits (`logit_scale = 1 / temperature`). | `0.005, 0.01, 0.02, 0.05` |

All default sweep lists also include the value in that dataset's YAML config.
The runner rejects invalid values, inactive loss paths, duplicate effective
memory capacities, and `neighbor_k` values that the model would silently clip.
It also rejects memory sweeps that would change the effective neighbor count,
so the requested factor remains the only changing factor.

Memory capacity is computed by the training code as
`min(round(samples_per_epoch * memory_ratio), memory_size)` when
`memory_size > 0`; otherwise the ratio controls capacity by itself. Under
distributed sampling, `samples_per_epoch` is the per-rank sample count. The
runner currently requires `sampler: random` so its per-rank sample and memory
capacity calculations match the training loader.

`soft_weight` and `neighbor_weight` are separate controls: the former sets the
current-identity target value, and the latter sets the increment for each
retrieved neighbor. The retrieved neighbor count is capped by the effective
memory size in the model. Training logs now report both the requested and
effective neighbor count.

In the current forward pass, prototype neighbors are visited in a loop, and
the soft ID targets are reassigned on each iteration. The subsequent soft ID
loss therefore uses the last prototype's retrieved neighbors. The runner
reports sweeps for this behavior as implemented; changing it to aggregate
neighbors from all prototypes would define a different experiment and require
new training runs.

## Run a sweep

Run one parameter on one dataset, using the dataset YAML as the baseline:

```bash
python tools/run_sensitivity.py \
  --dataset FRS-CMR \
  --parameter cprod_alpha \
  --seeds 1 2 3
```

Use explicit values or run the registered parameters for one dataset:

```bash
python tools/run_sensitivity.py --dataset FRS-CMR --parameter mask_pro --values 0.2 0.3 0.4 0.5
python tools/run_sensitivity.py --dataset CCPRD --parameter all --seeds 1 2 3
```

For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/run_sensitivity.py \
  --dataset FRS-CMR --parameter neighbor_k --seeds 1 2 3 --nproc-per-node 2
```

Preview commands without starting training:

```bash
python tools/run_sensitivity.py --dataset FRS-CMR --parameter memory_ratio --dry-run
```

Each run uses `val` for checkpoint selection. Evaluate the selected checkpoint
on `test` separately after choosing settings. Results are written under
`outputs/sensitivity/<dataset>/<session-id>/`. `summary.csv` and
`summary.json` contain per-seed records; `summary_by_value.csv` reports the
mean and standard deviation across completed seeds for each parameter value.

## Arguments excluded from the sweep

The parser's `alpha` and `beta` are Adam's optimizer betas, not model-loss
weights. `masked_token_rate` and `masked_token_unchanged_rate` are currently
accepted legacy config fields, but no active training path reads them. The
sensitivity runner therefore does not sweep these arguments.
