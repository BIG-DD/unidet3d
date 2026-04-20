#!/usr/bin/env bash
# Run tools/test.py inside Docker. Requires Docker and NVIDIA Container Toolkit.
#
# Usage:
#   ./run_docker_test.sh [config] [checkpoint] [extra args...]
#
# Example:
#   ./run_docker_test.sh configs/unidet3d_1xb8_scannet.py work_dirs/unidet3d_1xb8_scannet/latest.pth
#   ./run_docker_test.sh configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1024.pth --show-dir work_dirs/vis
set -e
cd "$(dirname "$0")"

if ! command -v docker &>/dev/null; then
  echo "Error: docker not found. Install with: sudo apt-get install -y docker.io"
  echo "For GPU support, also install NVIDIA Container Toolkit."
  exit 127
fi

CONFIG="${1:-configs/unidet3d_1xb8_scannet.py}"
CHECKPOINT="${2:-work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1024.pth}"
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

if ! docker image inspect unidet3d:latest &>/dev/null; then
  echo "Building image unidet3d:latest (this may take a long time)..."
  docker build -t unidet3d:latest .
fi

echo "Running: python tools/test.py $CONFIG $CHECKPOINT $*"
docker run --rm --gpus all \
  -v "$(pwd):/workspace" \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  unidet3d:latest \
  python tools/test.py "$CONFIG" "$CHECKPOINT" "$@"
