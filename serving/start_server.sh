#!/usr/bin/env bash
# Start Triton Inference Server on the course model repository.
#   bash serving/start_server.sh [extra tritonserver flags]
# Ports: HTTP 8000, gRPC 8001, Prometheus metrics 8002 (override with GK_HTTP_PORT etc.).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${GK_TRITON_HOME:-$HOME/.cache/gk_tritonserver}/env.sh"
export GK_ROOT="$(dirname "$HERE")"
export GK_CHECKPOINT="${GK_CHECKPOINT:-$GK_ROOT/checkpoints/hstu_ml-1m.pt}"
exec env LD_LIBRARY_PATH="$GK_TRITON_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$GK_TRITONSERVER" \
  --model-repository="$HERE/model_repository" \
  --backend-directory="$GK_TRITON_BACKENDS" \
  --http-port="${GK_HTTP_PORT:-8000}" --grpc-port="${GK_GRPC_PORT:-8001}" --metrics-port="${GK_METRICS_PORT:-8002}" \
  "$@"
