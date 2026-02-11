#!/usr/bin/env bash
# Build the Polaris Docker image (GPU + GUI support).
# Run from repo root: ./build.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${IMAGE_NAME:-polaris}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "Done. Image: ${IMAGE_NAME}:${IMAGE_TAG}"
