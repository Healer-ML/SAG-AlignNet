#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-CCPRD}"
processes="${2:-1}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
config="configs/agrisearch/${dataset}.yaml"
if [[ ! -f "$config" ]]; then
  echo "Unknown AgriSearch dataset config: $config" >&2
  echo "Choose one of FRS-CMR, CroDis4CMR, CCPRD, GjPest4CMR, FruDis4CMR." >&2
  exit 2
fi

if (( processes > 1 )); then
  exec torchrun --standalone --nproc_per_node="$processes" train.py --config-file "$config" "$@"
fi
exec python train.py --config-file "$config" "$@"
