#!/usr/bin/env bash
set -Eeuo pipefail

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.root_dir=/workspace \
  --ServerApp.allow_remote_access=True \
  --IdentityProvider.token="${JUPYTER_TOKEN:?JUPYTER_TOKEN is required}"
