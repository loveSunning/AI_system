# GPU 目标配置

这个仓库把两块重点显卡抽象成统一的 `AI_SYSTEM_GPU_PROFILE`：

| GPU | 架构 | Compute Capability | CMake 架构值 |
| --- | --- | --- | --- |
| RTX 4090 | Ada Lovelace | 8.9 | `89` |
| RTX 5060 | Blackwell | 12.0 | `120` |

## 配置方式

```text
-DAI_SYSTEM_GPU_PROFILE=native
-DAI_SYSTEM_GPU_PROFILE=4090
-DAI_SYSTEM_GPU_PROFILE=5060
-DAI_SYSTEM_GPU_PROFILE=all
```

## 说明

- `native` 会调用 `nvidia-smi --query-gpu=name,compute_cap` 自动探测本机 GPU。
- `all` 适合做一套工程同时覆盖 4090 和 5060 的二进制。
- 本仓库在 `2026-04-15` 的本机环境上已验证 `CUDA 13.0 + RTX 5060 (sm_120)` 的配置路径。
