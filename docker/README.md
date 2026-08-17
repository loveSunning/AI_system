# CuTe DSL CUDA 12.9 开发容器

这套配置面向 NVIDIA GeForce RTX 4090 D（Ada，Compute Capability 8.9），包含：

- Ubuntu 22.04
- CUDA Toolkit 12.9.2（含 `nvcc` 和开发头文件）
- PyTorch 2.8.0 + CUDA 12.9
- NVIDIA CUTLASS / CuTe DSL 稳定版
- JupyterLab、CMake、Ninja、Git、pytest 和 mypy

目录结构：

```text
docker/
├── dockerfile/
│   └── Dockerfile.cutedsl_dev
├── container/
│   ├── .env.example
│   └── docker-compose-cutedsl-dev.yaml
└── README.md
```

## 1. 宿主机准备

需要：

1. NVIDIA RTX 4090 D 驱动；CUDA 12.9 对应的 Linux 驱动最低为 575.51.03。
2. Docker Engine + Docker Compose v2。
3. NVIDIA Container Toolkit，用于把宿主机 NVIDIA GPU 暴露给容器。

先在宿主机检查：

```bash
nvidia-smi
docker version
docker compose version
```

验证 Docker 可以访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu22.04 nvidia-smi
```

如果这里无法看到 4090 D，请先修复驱动或 NVIDIA Container Toolkit，后续构建容器也不会解决宿主机 GPU 接入问题。

## 2. 创建本地 Compose 参数文件

进入 compose 目录（请把 `/path/to/AI_system` 替换为仓库实际路径）：

```bash
cd /path/to/AI_system/docker/container
cp .env.example .env
```

默认 `WORKSPACE_PATH=../..`，会把本仓库根目录挂载到容器的 `/workspace`。如果要挂载其他目录，请编辑 `.env`，例如：

```dotenv
WORKSPACE_PATH=/home/user/workspace/AI_system
```

把 `.env` 中的 `USER_ID` 和 `GROUP_ID` 改为以下命令的输出，以避免 bind mount 文件权限问题：

```bash
id -u
id -g
```

## 3. 构建镜像

在 `docker/container` 目录执行：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml build
```

首次构建需要下载 CUDA 基础镜像、PyTorch 和 CuTe DSL，耗时取决于网络速度。

强制拉取最新的同名基础镜像并重新构建：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml build --pull --no-cache
```

## 4. 启动容器

后台启动：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml up -d
```

查看状态：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml ps
```

进入容器：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml exec cutedsl_dev bash
```

默认 Python 虚拟环境位于 `/opt/cutedsl-venv`，已经加入 `PATH`，无需手动激活。

## 5. 验证 CUDA、PyTorch 和 CuTe DSL

进入容器后执行：

```bash
nvcc --version
nvidia-smi
```

再执行：

```bash
python - <<'PY'
import torch
import cutlass
from cutlass import cute

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("CuTe DSL import: OK")
PY
```

4090 D 的关键输出应类似：

```text
PyTorch CUDA: 12.9
CUDA available: True
GPU: NVIDIA GeForce RTX 4090 D
Compute capability: (8, 9)
CuTe DSL import: OK
```

## 6. 使用 CUTLASS 官方示例

NVIDIA 建议让 GitHub 源码与 DSL 包版本保持一致。需要最新仓库示例时，可在 `/workspace/3rdparty` 下克隆 CUTLASS，再使用对应提交中的安装脚本：

```bash
cd /workspace/3rdparty
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
./python/CuTeDSL/setup.sh --cu12
```

仓库中的部分 Hopper、Blackwell 示例只支持 SM90/SM100，不会在 4090 D（SM89）上运行。请选择标明支持 Ampere/Ada 或 SM80/SM89 的示例。

## 7. 启动 JupyterLab（可选）

容器内执行：

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

默认 compose 没有发布端口。需要从宿主机浏览器访问时，在 `cutedsl_dev` 服务中加入：

```yaml
ports:
  - "8888:8888"
```

然后重新创建容器：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml up -d --force-recreate
```

## 8. 停止、重启和删除容器

停止：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml stop
```

重新启动：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml start
```

删除容器和 Compose 网络，但保留构建好的镜像及宿主机工作区文件：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml down
```

代码位于宿主机 bind mount 中，`down` 不会删除 `/workspace` 对应的宿主机文件。

## 9. 常见问题

### `could not select device driver` 或容器看不到 GPU

Docker 尚未正确接入 NVIDIA GPU。先确保第 1 步的 `docker run --gpus all ... nvidia-smi` 成功。

### `CUDA driver version is insufficient`

升级宿主机 NVIDIA 驱动。容器共享宿主机驱动，镜像中的 CUDA Toolkit 不能替代宿主机驱动。

### 修改了 `USER_ID`、`GROUP_ID` 或 Dockerfile，但容器没有变化

重新构建并创建容器：

```bash
docker compose -f docker-compose-cutedsl-dev.yaml up -d --build --force-recreate
```

### PyTorch 显示 CUDA 12.9，但 `nvidia-smi` 显示更高 CUDA 版本

这是正常的。`nvidia-smi` 显示驱动最高支持的 CUDA 版本，`torch.version.cuda` 显示 PyTorch wheel 的构建版本，`nvcc --version` 显示容器 Toolkit 版本。

### 为什么没有使用 `privileged: true` 和整机 `/dev` 挂载

CuTe DSL 只需要 NVIDIA GPU 设备。Compose 已通过 NVIDIA device reservation 暴露 GPU，无需给容器完整宿主机设备权限。
