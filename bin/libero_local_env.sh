#!/usr/bin/env bash

# Local runtime defaults for LIBERO reproduction without Docker.
# Source from any directory:
#   source bin/libero_local_env.sh

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ -d "${REPO_ROOT}/.venv" ]; then
  export VIRTUAL_ENV="${REPO_ROOT}/.venv"
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_DISABLE_XET="1"
export MUJOCO_GL="egl"
export UV_LINK_MODE="copy"

if [ -z "${BASE_DATASETS_DIR:-}" ]; then
  export BASE_DATASETS_DIR="${REPO_ROOT}"
fi
