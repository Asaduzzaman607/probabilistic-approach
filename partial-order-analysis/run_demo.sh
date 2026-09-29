#!/usr/bin/env bash
# One-command setup + demo run. From the repository root:
#   bash run_demo.sh
#
# Creates/reuses a local venv, installs dependencies, and runs the one
# script in this repository that executes end-to-end using only what's
# included here (real sample data + fabricated student attempts -- see
# code/demo_synthetic_run.py and README.md for details).
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -r requirements.txt

cd code
python3 demo_synthetic_run.py
