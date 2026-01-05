#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCH="${ACT_ARCH:-linux/amd64}"
UBUNTU_TAG="${ACT_UBUNTU_TAG:-act-24.04}"
UBUNTU_IMAGE="ghcr.io/catthehacker/ubuntu:${UBUNTU_TAG}"
PLATFORM="ubuntu-24.04=${UBUNTU_IMAGE}"
DEFAULT_EVENTS=("pull_request" "push" "workflow_dispatch")
WORKFLOWS_DIR="${ACT_WORKFLOWS_DIR:-.github/workflows}"

# This override must be a path that the Docker daemon can see, not a macOS client socket path.
DOCKER_DAEMON_SOCKET_PATH="${ACT_DOCKER_DAEMON_SOCKET_PATH:-/var/run/docker.sock}"

if ! command -v act >/dev/null 2>&1; then
  echo "ERROR: act is not installed"
  echo "Install with: brew install act"
  exit 1
fi

if [ ! -d "$WORKFLOWS_DIR" ]; then
  echo "ERROR: workflows dir not found: $WORKFLOWS_DIR"
  echo "Run from repo root, or set ACT_WORKFLOWS_DIR"
  exit 1
fi

CONTAINER_DOCKER_SOCK="/var/run/docker.sock"
CONTAINER_OPTIONS="${ACT_CONTAINER_OPTIONS:-}"

SOCKET_MOUNT_OPT="-v ${DOCKER_DAEMON_SOCKET_PATH}:${CONTAINER_DOCKER_SOCK}"
if echo " ${CONTAINER_OPTIONS} " | grep -q " ${CONTAINER_DOCKER_SOCK}"; then
  SOCKET_MOUNT_OPT=""
fi

EVENT="${1:-all}"
shift || true
EXTRA_ARGS=("$@")

run_event() {
  local ev="$1"
  echo
  echo "=============================="
  echo " Running act event: ${ev}"
  echo " workflows: ${WORKFLOWS_DIR}"
  echo " platform: ${PLATFORM}"
  echo " arch: ${ARCH}"
  echo " daemon socket mount: ${DOCKER_DAEMON_SOCKET_PATH} -> ${CONTAINER_DOCKER_SOCK}"
  echo "=============================="

  local act_args=(
    "${ev}"
    -W "${WORKFLOWS_DIR}"
    --container-architecture "${ARCH}"
    -P "${PLATFORM}"
    --container-daemon-socket "-"   # disable act default socket bind mount
  )

  # Important, this is what makes docker inside the runner work
  if [ -n "${SOCKET_MOUNT_OPT}" ] || [ -n "${CONTAINER_OPTIONS}" ]; then
    act_args+=( --container-options "${SOCKET_MOUNT_OPT} ${CONTAINER_OPTIONS}" )
  fi

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    act "${act_args[@]}" "${EXTRA_ARGS[@]}"
  else
    act "${act_args[@]}"
  fi
}

if [ "${EVENT}" = "all" ]; then
  for ev in "${DEFAULT_EVENTS[@]}"; do
    run_event "${ev}"
  done
else
  run_event "${EVENT}"
fi
