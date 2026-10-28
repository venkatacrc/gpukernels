#!/usr/bin/env bash
# Install NVIDIA Triton Inference Server without Docker, for the Python backend.
#
# Where Docker is available, use the official image instead and skip this file:
#   docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
#     -v $PWD:/course nvcr.io/nvidia/tritonserver:<yy.mm>-py3 \
#     tritonserver --model-repository=/course/serving/model_repository
#
# Without Docker (this machine), the nvidia-pytriton wheel is the easiest way to
# get a real tritonserver binary: it bundles Triton 2.51 for x86_64 and aarch64.
# Three things need fixing to run its Python backend outside PyTriton:
#
#  1. The bundled libraries are found through a relative RPATH, so the server
#     is run in place with LD_LIBRARY_PATH pointing at nvidia_pytriton.libs.
#  2. The backend stub embeds a CPython 3.13.0 (built with pyenv under
#     /root/.pyenv/versions/3.13.0) and initialises it in *isolated* mode, which
#     ignores PYTHONHOME and PYTHONPATH. It must find a matching 3.13.0 standard
#     library at that compiled-in prefix. We install python=3.13.0 from
#     conda-forge at a path of exactly the same length and rewrite the prefix
#     string in a copy of libpython.
#  3. Isolated mode also ignores PYTHONPATH, so a .pth file in that prefix's
#     site-packages adds the course venv (torch, triton, numpy) to sys.path.
#
# Usage:  bash serving/setup_tritonserver.sh          (needs network, conda, pip)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${GK_PYTHON:-$ROOT/../bin/python}"
GK_TRITON_HOME="${GK_TRITON_HOME:-$HOME/.cache/gk_tritonserver}"
PYTRITON_VERSION=0.7.0
ORIG_PREFIX="/root/.pyenv/versions/3.13.0"

pick_prefix() {
  # a writable path with exactly ${#ORIG_PREFIX} characters
  local want=${#ORIG_PREFIX} base
  for base in "$HOME/.cache/tpy" "$HOME/.tpy" "/tmp/tpy"; do
    if (( ${#base} <= want )); then
      local p="$base"
      while (( ${#p} < want )); do p="${p}x"; done
      echo "$p"
      return
    fi
  done
  echo "no writable prefix of length $want; set GK_STUB_PREFIX" >&2
  exit 1
}
STUB_PREFIX="${GK_STUB_PREFIX:-$(pick_prefix)}"
[[ ${#STUB_PREFIX} -eq ${#ORIG_PREFIX} ]] || { echo "GK_STUB_PREFIX must be ${#ORIG_PREFIX} characters"; exit 1; }

echo "== 1/5 tritonserver from nvidia-pytriton $PYTRITON_VERSION -> $GK_TRITON_HOME/server"
mkdir -p "$GK_TRITON_HOME"
if [[ ! -x "$GK_TRITON_HOME/server/pytriton/tritonserver/bin/tritonserver" ]]; then
  "$PY" -m pip install --quiet --no-deps --target "$GK_TRITON_HOME/server" "nvidia-pytriton==$PYTRITON_VERSION"
fi
LIBS="$GK_TRITON_HOME/server/nvidia_pytriton.libs"
TS="$GK_TRITON_HOME/server/pytriton/tritonserver"

echo "== 2/5 tritonclient -> $GK_TRITON_HOME/client"
if [[ ! -d "$GK_TRITON_HOME/client/tritonclient" ]]; then
  "$PY" -m pip install --quiet --target "$GK_TRITON_HOME/client" "tritonclient[http,grpc]>=2.60"
  # keep the venv's numpy: a second copy in the target dir would shadow it
  rm -rf "$GK_TRITON_HOME/client"/numpy "$GK_TRITON_HOME/client"/numpy-* "$GK_TRITON_HOME/client"/numpy.libs
fi

echo "== 3/5 CPython 3.13.0 standard library at $STUB_PREFIX"
if [[ ! -x "$STUB_PREFIX/bin/python3.13" ]]; then
  conda create -y -q -p "$STUB_PREFIX" -c conda-forge --override-channels python=3.13.0 >/dev/null
fi

echo "== 4/5 point the stub's libpython at that prefix"
LIBPY="$(ls "$LIBS"/libpython3-*.so.1.0 | head -1)"
[[ -f "$LIBPY.orig" ]] || cp "$LIBPY" "$LIBPY.orig"
"$PY" - "$LIBPY" "$ORIG_PREFIX" "$STUB_PREFIX" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2].encode(), sys.argv[3].encode()
data = open(path + ".orig", "rb").read()
assert len(old) == len(new) and data.count(old) > 0
open(path, "wb").write(data.replace(old, new))
print(f"   rewrote {data.count(old)} occurrence(s) of {old.decode()}")
EOF
cp "$TS/python_backend_stubs/3.13/triton_python_backend_stub" "$TS/backends/python/"

echo "== 5/5 make the venv importable inside the stub"
VENV_SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n%s\n' "$VENV_SP" "$GK_TRITON_HOME/client" "$ROOT" > "$STUB_PREFIX/lib/python3.13/site-packages/gk_course.pth"

cat > "$GK_TRITON_HOME/env.sh" <<EOF
export GK_TRITON_HOME="$GK_TRITON_HOME"
export GK_TRITONSERVER="$TS/bin/tritonserver"
export GK_TRITON_BACKENDS="$TS/backends"
export GK_TRITON_LIBS="$LIBS"
export GK_TRITON_CLIENT="$GK_TRITON_HOME/client"
EOF
echo "done. Start the server with: bash serving/start_server.sh"
