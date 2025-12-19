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

Build and run DGL on NVIDIA DGX Spark with Blackwell GPU. **Verified working** with:
- DGL Core, DGL Sparse, and DGL GraphBolt
- CUDA 12.8, PyTorch 2.6, sm_120 architecture
- Build time: ~6 minutes

```bash
# Build image (uses nvcr.io/nvidia/pytorch:25.01-py3 base)
./build_dgx_spark.sh build

# Verify installation
./build_dgx_spark.sh run

# Run tests
./build_dgx_spark.sh test

# Interactive shell
./build_dgx_spark.sh shell
```

See [Dockerfile.dgx_spark](Dockerfile.dgx_spark) and [docs/DGX_SPARK.md](../docs/DGX_SPARK.md) for details.

