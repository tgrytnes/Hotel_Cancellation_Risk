#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/exp_baseline.yaml}
python -m hcr.preprocess --config $CONFIG
python -m hcr.features --config $CONFIG
python -m hcr.train --config $CONFIG
python -m hcr.eval --config $CONFIG
echo "Done."
