#!/bin/bash
# Build the MatClaw agent container image
#
# Usage:
#   ./build.sh              # Build CPU image (matclaw-agent:latest)
#   ./build.sh --cuda       # Build CUDA/GPU image (matclaw-agent:cuda)
#   ./build.sh v1.0         # Build CPU image with custom tag
#   ./build.sh v1.0 --cuda  # Build CUDA/GPU image with custom tag

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="matclaw-agent"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
TAG=""
CUDA=false

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    --cuda)
      CUDA=true
      ;;
    *)
      TAG="$arg"
      ;;
  esac
done

BUILD_ARGS=""

if [ "$CUDA" = true ]; then
  TAG="${TAG:-cuda}"
  BUILD_ARGS="--build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
  BUILD_ARGS="$BUILD_ARGS --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu128"
  echo "Building MatClaw agent container image (CUDA/GPU)..."
else
  TAG="${TAG:-latest}"
  echo "Building MatClaw agent container image (CPU)..."
fi

echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""

${CONTAINER_RUNTIME} build ${BUILD_ARGS} -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${TAG}"

if [ "$CUDA" = true ]; then
  echo ""
  echo "To use GPU containers, set in .env:"
  echo "  CONTAINER_GPU=true"
  echo "  CONTAINER_IMAGE=matclaw-agent:cuda"
  echo ""
  echo "Test with:"
  echo "  echo '{\"prompt\":\"What GPU is available?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run --gpus all -i ${IMAGE_NAME}:${TAG}"
else
  echo ""
  echo "Test with:"
  echo "  echo '{\"prompt\":\"What is 2+2?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run -i ${IMAGE_NAME}:${TAG}"
fi
