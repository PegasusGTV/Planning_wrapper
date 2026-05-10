#!/usr/bin/env bash
# Run the Polaris container in the terminal. Persistent container: same state when you exit and run again.
# Mounts current directory as /workspace, GPU and X11 enabled. Run from repo root: ./run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-polaris-dev}"
IMAGE_NAME="${IMAGE_NAME:-polaris}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"

# X11 for GUI (use host DISPLAY if set)
DISPLAY="${DISPLAY:-:0}"
xhost +local:docker 2>/dev/null || true

_run_shell() {
  exec docker exec -it -e DISPLAY="${DISPLAY}" "${CONTAINER_NAME}" /bin/bash
}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already running. Opening shell..."
    _run_shell
  else
    echo "Starting existing container ${CONTAINER_NAME}..."
    docker start "${CONTAINER_NAME}"
    _run_shell
  fi
else
  echo "Creating container ${CONTAINER_NAME} (image: ${IMAGE_NAME}:${IMAGE_TAG})..."
  docker run -d --name "${CONTAINER_NAME}" \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e LD_LIBRARY_PATH=/host-lib:/host-lib-usr \
    -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd_usr.json \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /usr/lib/x86_64-linux-gnu:/host-lib:ro \
    -v /usr/lib:/host-lib-usr:ro \
    -v "${WORKSPACE}:/workspace:rw" \
    -w /workspace \
    "${IMAGE_NAME}:${IMAGE_TAG}"
  echo "Opening shell (exit to leave container running; use ./terminal.sh to reconnect)..."
  _run_shell
fi
