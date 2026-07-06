# GPU Targets

The root project uses `AI_SYSTEM_GPU_PROFILE` to map the two supported lab environments to CMake CUDA architectures.

| Environment / preset | `AI_SYSTEM_GPU_PROFILE` | CUDA architecture | Intended GPU |
| --- | --- | --- |
| Windows: `windows-vs2022-cuda-release` | `5060` | `120` | GeForce RTX 5060 |
| Linux / WSL: `linux-make-cuda-release` | `4090d` | `89` | GeForce RTX 4090D |

The root CMake code still accepts `native`, `4090d`, `5060`, or a manual semicolon-separated SM list for experiments, but the checked-in presets intentionally expose only one CUDA GPU target per OS.

NVIDIA's CUDA GPU compute capability table lists RTX 5060 under compute capability `12.0`, which corresponds to CMake `CMAKE_CUDA_ARCHITECTURES=120`. The same table lists RTX 4090-class Ada GPUs under compute capability `8.9`, which corresponds to `CMAKE_CUDA_ARCHITECTURES=89`.

CUTLASS official docs show examples for architecture values such as `90a`, `100a`, `80`, and `75`. For this lab, use `120` on Windows and `89` on Linux. If the installed CUDA Toolkit does not accept the expected SM, upgrade the toolkit.

Recommended local checks:

```powershell
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version
nvcc --list-gpu-code
Test-Path "$env:CUDA_PATH\include\cutlass\cutlass.h"
Test-Path "$env:CUDA_PATH\include\cute\tensor.hpp"
cmake --version
```

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version
nvcc --list-gpu-code
test -f "${CUDA_HOME:-/usr/local/cuda}/include/cutlass/cutlass.h"
test -f "${CUDA_HOME:-/usr/local/cuda}/include/cute/tensor.hpp"
cmake --version
```

References:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
