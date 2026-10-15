# SAG-AlignNet inference checkpoints

Five model-only checkpoints (one per AgriSearch dataset) are stored with Git LFS. Each file contains a `model` state dictionary plus the selected epoch metadata. Optimizer and scheduler states are omitted; these artifacts support inference/evaluation, not resume-training.

| File | Dataset | Config |
| --- | --- | --- |
| `FRS-CMR.pth` | FRS-CMR | `configs/agrisearch/FRS-CMR.yaml` |
| `CroDis4CMR.pth` | CroDis4CMR | `configs/agrisearch/CroDis4CMR.yaml` |
| `CCPRD.pth` | CCPRD | `configs/agrisearch/CCPRD.yaml` |
| `GjPest4CMR.pth` | GjPest4CMR | `configs/agrisearch/GjPest4CMR.yaml` |
| `FruDis4CMR.pth` | FruDis4CMR | `configs/agrisearch/FruDis4CMR.yaml` |

See `metadata.json` for the source run and evaluation-selection split. The historical source runs selected `best.pth` on test; this provenance is retained here so the artifacts are not mistaken for validation-selected results.
