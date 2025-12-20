# DGL on NVIDIA DGX Spark

Build and run DGL on NVIDIA DGX Spark with Grace Blackwell (GB10) GPU.

## Verified Build ✅

| Component | CUDA 12.8 | CUDA 13.0 |
|-----------|-----------|-----------|
| DGL Core | ✅ | ✅ |
| DGL Sparse | ✅ | ✅ |
| DGL GraphBolt | ✅ | ✅ |
| GNN Layers | ✅ | ✅ |

## Quick Start

```bash
cd docker

# Build (CUDA 13 is default)
./build_dgx_spark.sh build

# Verify installation
./build_dgx_spark.sh run

# Run tests
./build_dgx_spark.sh test

# Interactive shell
./build_dgx_spark.sh shell
```

## CUDA Versions

| Version | Base Image | PyTorch | CUDA | Architectures |
|---------|------------|---------|------|---------------|
| cuda12 | `nvcr.io/nvidia/pytorch:25.01-py3` | 2.6 | 12.8 | sm_120 |
| cuda13 | `nvcr.io/nvidia/pytorch:25.11-py3` | 2.10 | 13.0 | sm_120, sm_121 |

```bash
# CUDA 13 (default)
./build_dgx_spark.sh cuda13 build
./build_dgx_spark.sh cuda13 run

# CUDA 12
./build_dgx_spark.sh cuda12 build
./build_dgx_spark.sh cuda12 run
```

## Requirements

- NVIDIA DGX Spark with GB10 GPU
- Docker with NVIDIA Container Toolkit
- Network access to NGC (`nvcr.io`)

## Build Options

Environment variables for customization:

```bash
# Parallel build jobs (default: 8)
MAX_JOBS=16 ./build_dgx_spark.sh build
```

## Manual Build

If you prefer to build without Docker:

```bash
# Set environment
export CUDA_ARCH_NAME=Blackwell
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export CUDAARCHS="120;121"

# Build
mkdir build && cd build
cmake -DUSE_CUDA=ON \
      -DCUDA_ARCH_NAME=Blackwell \
      -DBUILD_SPARSE=ON \
      -DBUILD_GRAPHBOLT=ON \
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
nvcc --version
```
