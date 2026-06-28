#!/usr/bin/env bash
set -euo pipefail

python -m cot_research.experiment_runner --config configs/cot_smoke_mock.json
