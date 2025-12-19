# DGL on NVIDIA DGX Spark

Build and run DGL on NVIDIA DGX Spark with Grace Blackwell (GB10) GPU.

## Verified Build ✅

Build tested on December 19, 2025:

| Component | Status |
|-----------|--------|
| DGL Core | ✅ Working |
| DGL Sparse | ✅ Working (BUILD_SPARSE=ON) |
| DGL GraphBolt | ✅ Working (BUILD_GRAPHBOLT=ON) |
| Build Time | ~6 minutes |

### Tested Functionality
- Graph creation and GPU transfer
- Sparse matrix operations (SpMV, transpose)
- GraphBolt ItemSet and FusedCSCSamplingGraph

## Quick Start

```bash
cd docker

# Build the Docker image
./build_dgx_spark.sh build

# Verify installation
./build_dgx_spark.sh run

# Run tests
./build_dgx_spark.sh test

# Interactive shell
./build_dgx_spark.sh shell
```

## Requirements

- NVIDIA DGX Spark with GB10 GPU
- Docker with NVIDIA Container Toolkit
- Network access to NGC (`nvcr.io`)

## Architecture

| Component | Version |
|-----------|---------|
| Base Image | `nvcr.io/nvidia/pytorch:25.01-py3` |
| CUDA | 12.8 |
| PyTorch | 2.6 |
| GPU Arch | sm_120 (Blackwell) |

## Build Options

Environment variables for customization:

```bash
# Parallel build jobs (default: 8)
MAX_JOBS=16 ./build_dgx_spark.sh build

# Custom CUDA architecture
CUDA_ARCH=121 ./build_dgx_spark.sh build
```

## Manual Build

If you prefer to build without Docker:

```bash
# Set environment
export CUDA_ARCH_NAME=Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"

# Build
mkdir build && cd build
cmake -DUSE_CUDA=ON \
      -DCUDA_ARCH_NAME=Blackwell \
      -DBUILD_TYPE=release \
      ..
make -j$(nproc)

# Install
cd ../python && pip install -e .
```

## Known Limitations

1. **LIBXSMM**: Disabled on ARM64 (x86-only optimization)
2. **NVIDIA Deprecation**: DGL containers deprecated after 25.08; consider PyG for new projects

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-ctk --version
```

### Build Failures

Check CUDA version compatibility:
```bash
nvcc --version  # Should be 12.8+
```
