## Build docker image for CI

### CPU image
```bash
docker build -t dgl-cpu -f Dockerfile.ci_cpu .
```

### GPU image
```bash
docker build -t dgl-gpu -f Dockerfile.ci_gpu .
```

### Lint image
```bash
docker build -t dgl-lint -f Dockerfile.ci_lint .
```

### CPU image for kg
```bash
wget https://data.dgl.ai/dataset/FB15k.zip -P install/
docker build -t dgl-cpu:torch-1.2.0 -f Dockerfile.ci_cpu_torch_1.2.0 .
```

### GPU image for kg
```bash
wget https://data.dgl.ai/dataset/FB15k.zip -P install/
docker build -t dgl-gpu:torch-1.2.0 -f Dockerfile.ci_gpu_torch_1.2.0 .
```

### DGX Spark (Grace Blackwell GB10) ✅

Build and run DGL on NVIDIA DGX Spark with Blackwell GPU.

#### Quick Start
```bash
# Build and run (defaults to CUDA 13)
./build_dgx_spark.sh build
./build_dgx_spark.sh run
./build_dgx_spark.sh test
```

#### CUDA Versions

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

# Interactive shell
./build_dgx_spark.sh cuda13 shell
```

See [docs/DGX_SPARK.md](../docs/DGX_SPARK.md) for more details.
