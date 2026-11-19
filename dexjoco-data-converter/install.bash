#!/usr/bin/env bash
set -euo pipefail

git submodule update --init -- dexjoco-data-converter/src/3rd/diffusion_policy
conda env create -f ./dexjoco-data-converter/environment-dc.yaml
conda run --no-capture-output -n dexjoco-dc python -m pip install -e ./dexjoco-data-converter
