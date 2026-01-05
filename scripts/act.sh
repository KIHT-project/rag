#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ===== Defaults =====
ARCH="${ACT_ARCH:-linux/amd64}"
UBUNTU_TAG="${ACT_UBUNTU_TAG:-act-24.04}"
UBUNTU_IMAGE="ghcr.io/catthehacker/ubuntu:${UBUNTU_TAG}"
PLATFORM="ubuntu-24.04=${UBUNTU_IMAGE}"
DEFAULT_EVENTS=("pull_request" "push" "workflow_dispatch")

# ===== Checks =====
if ! command -v act >/dev/null 2>&1; then
  echo "ERROR: act is not installed"
  echo "Install with: brew install act"
  exit 1
fi

# ===== Args =====
EVENT="${1:-all}"
shift || true
EXTRA_ARGS=("$@")

run_event() {
  local ev="$1"
  echo
  echo "=============================="
  echo " Running act event: ${ev}"
  echo "=============================="

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    act "${ev}" \
      --container-architecture "${ARCH}" \
      -P "${PLATFORM}" \
      "${EXTRA_ARGS[@]}"
  else
    act "${ev}" \
      --container-architecture "${ARCH}" \
      -P "${PLATFORM}"
  fi
}

# ===== Dispatch =====
if [[ "${EVENT}" == "all" ]]; then
  for ev in "${DEFAULT_EVENTS[@]}"; do
    run_event "${ev}"
  done
else
  run_event "${EVENT}"
fi
