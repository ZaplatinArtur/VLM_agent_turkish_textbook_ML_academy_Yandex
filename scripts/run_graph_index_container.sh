#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --quiet --disable-pip-version-check \
  faiss-cpu "sentence-transformers>=3.0,<5.4"

exec python3 -m retrieve.build_index \
  --sample-query dikdortgen \
  --k 3
