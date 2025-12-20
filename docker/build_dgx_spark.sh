#!/bin/bash
# Build DGL for NVIDIA DGX Spark (Grace Blackwell GB10)
# Usage: ./build_dgx_spark.sh [cuda12|cuda13] [build|run|test|shell]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGL_ROOT="$(dirname "$SCRIPT_DIR")"

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
    echo "Usage: $0 [cuda12|cuda13] [command]"
    echo ""
    echo "CUDA Versions:"
    echo "  cuda12    CUDA 12.8 with PyTorch 2.6 (NGC 25.01)"
    echo "  cuda13    CUDA 13.0 with PyTorch 2.10 (NGC 25.11) [default]"
    echo ""
    echo "Commands:"
    echo "  build   Build the Docker image (default)"
    echo "  run     Run verification tests"
    echo "  test    Run comprehensive DGL tests"
    echo "  shell   Open interactive shell in container"
    echo "  help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 cuda13 build    # Build CUDA 13 image"
    echo "  $0 cuda12 run      # Run CUDA 12 verification"
    echo "  $0 build           # Build default (CUDA 13)"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_JOBS    Number of parallel build jobs (default: 8)"
}

# Parse CUDA version
parse_cuda_version() {
    case "$1" in
        cuda12)
            CUDA_VERSION="cuda12"
            DOCKERFILE="Dockerfile.dgx_spark_cuda12"
            BASE_IMAGE="nvcr.io/nvidia/pytorch:25.01-py3"
            IMAGE_NAME="dgl:cuda12"
            shift
            ;;
        cuda13)
            CUDA_VERSION="cuda13"
            DOCKERFILE="Dockerfile.dgx_spark_cuda13"
            BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
            IMAGE_NAME="dgl:cuda13"
            shift
            ;;
        build|run|test|shell|help|--help|-h)
            # Default to CUDA 13 if no version specified
            CUDA_VERSION="cuda13"
            DOCKERFILE="Dockerfile.dgx_spark_cuda13"
            BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
            IMAGE_NAME="dgl:cuda13"
            ;;
        *)
            # Default to CUDA 13
            CUDA_VERSION="cuda13"
            DOCKERFILE="Dockerfile.dgx_spark_cuda13"
            BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
            IMAGE_NAME="dgl:cuda13"
            ;;
    esac
}

do_build() {
    print_status "Building DGL Docker image (${CUDA_VERSION})..."
    print_status "Using base: ${BASE_IMAGE}"
    
    cd "$DGL_ROOT"
    
    DOCKER_BUILDKIT=1 docker build \
        --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
        -f "docker/${DOCKERFILE}" \
        -t "${IMAGE_NAME}" \
        .
    
    print_status "Build complete: ${IMAGE_NAME}"
}

do_run() {
    print_status "Running DGL verification (${CUDA_VERSION})..."
    
    docker run --rm --gpus all --ipc=host \
        "${IMAGE_NAME}"
}

do_test() {
    print_status "Running DGL comprehensive tests (${CUDA_VERSION})..."
    
    docker run --rm --gpus all --ipc=host \
        "${IMAGE_NAME}" \
        python3 -c "
import dgl
import torch
import dgl.function as fn
from dgl.nn import GraphConv, GATConv, SAGEConv

print('=== DGL ${CUDA_VERSION^^} Test Suite ===')
print()

# Test 1: Graph operations
g = dgl.graph(([0,1,2,3,4], [1,2,3,4,0])).to('cuda')
print('[1/6] Graph on CUDA:', g.device, '✓')

# Test 2: Heterogeneous graphs
hg = dgl.heterograph({('u', 'e', 'v'): ([0,1], [1,2])}).to('cuda')
print('[2/6] Heterogeneous graph ✓')

# Test 3: Message passing
g.ndata['h'] = torch.randn(5, 64, device='cuda')
g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
print('[3/6] Message passing ✓')

# Test 4: GNN layers
gcn = GraphConv(64, 32).to('cuda')
out = gcn(g, g.ndata['h'])
print('[4/6] GraphConv layer ✓')

# Test 5: Batching
graphs = [dgl.rand_graph(10, 30).to('cuda') for _ in range(10)]
batched = dgl.batch(graphs)
print('[5/6] Graph batching ✓')

# Test 6: DGL Sparse
from dgl import sparse as dglsp
idx = torch.tensor([[0,1],[1,0]], device='cuda')
val = torch.tensor([1.,2.], device='cuda')
A = dglsp.spmatrix(idx, val, shape=(2,2))
print('[6/6] DGL Sparse ✓')

print()
print('=== All Tests Passed! ===')
"
}

do_shell() {
    print_status "Opening interactive shell (${CUDA_VERSION})..."
    
    docker run --rm -it --gpus all --ipc=host \
        -v "${DGL_ROOT}:/workspace/dgl" \
        -w /workspace/dgl \
        "${IMAGE_NAME}" \
        /bin/bash
}

# Main
parse_cuda_version "$1"

# Determine command (could be $1 or $2)
if [[ "$1" == "cuda12" || "$1" == "cuda13" ]]; then
    COMMAND="${2:-build}"
else
    COMMAND="${1:-build}"
fi

case "${COMMAND}" in
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
        print_error "Unknown command: ${COMMAND}"
        show_help
        exit 1
        ;;
esac
