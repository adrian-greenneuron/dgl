#!/bin/bash
# Build DGL for NVIDIA DGX Spark (Grace Blackwell GB10)
# Usage: ./build_dgx_spark.sh [build|run|test|shell]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGL_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="dgl-dgx-spark"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[DGL-Spark]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "DGL Build Script for NVIDIA DGX Spark"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build   Build the Docker image (default)"
    echo "  run     Run verification tests"
    echo "  test    Run pytest tests"
    echo "  shell   Open interactive shell in container"
    echo "  help    Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_JOBS    Number of parallel build jobs (default: 8)"
    echo "  CUDA_ARCH   CUDA architecture (default: 120)"
}

do_build() {
    print_status "Building DGL Docker image for DGX Spark..."
    print_status "Using base: nvcr.io/nvidia/pytorch:25.01-py3"
    
    cd "$DGL_ROOT"
    
    docker build \
        --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
        --build-arg CUDA_ARCH="${CUDA_ARCH:-120}" \
        -f docker/Dockerfile.dgx_spark \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        .
    
    print_status "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
}

do_run() {
    print_status "Running DGL verification..."
    
    docker run --rm --gpus all \
        "${IMAGE_NAME}:${IMAGE_TAG}"
}

do_test() {
    print_status "Running DGL pytest tests..."
    
    docker run --rm --gpus all \
        -v "${DGL_ROOT}:/workspace/dgl" \
        -w /workspace/dgl \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3 -m pytest tests/python/pytorch/test_basics.py -v -x --timeout=300
}

do_shell() {
    print_status "Opening interactive shell..."
    
    docker run --rm -it --gpus all \
        -v "${DGL_ROOT}:/workspace/dgl" \
        -w /workspace/dgl \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        /bin/bash
}

# Main
case "${1:-build}" in
    build)
        do_build
        ;;
    run)
        do_run
        ;;
    test)
        do_test
        ;;
    shell)
        do_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
