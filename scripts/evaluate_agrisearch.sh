#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: bash scripts/evaluate_agrisearch.sh DATASET CHECKPOINT [val|test]}"
checkpoint="${2:?usage: bash scripts/evaluate_agrisearch.sh DATASET CHECKPOINT [val|test]}"
split="${3:-test}"
config="configs/agrisearch/${dataset}.yaml"
if [[ ! -f "$config" ]]; then
  echo "Unknown AgriSearch dataset config: $config" >&2
  exit 2
fi
python evaluate.py --config-file "$config" --checkpoint "$checkpoint" --split "$split"
