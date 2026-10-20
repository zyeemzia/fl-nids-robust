#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python tests/test_core.py
cd src
python data.py
python run_noniid.py
python run_poisoning.py
python report.py
