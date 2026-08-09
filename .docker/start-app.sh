#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${APP_COMMAND:-}" ]]; then
  echo "app: APP_COMMAND is empty; supervised application is disabled"
  exec sleep infinity
fi

echo "app: starting ${APP_COMMAND}"
exec bash -lc "exec ${APP_COMMAND}"
