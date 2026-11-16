# vLLM 0.33.0 CUDA 13 开发容器运行指南

核对时间：2026-11-16。目标主机：`cdky@192.168.88.91`，GPU RTX 4090 D，驱动 580.82.09。本次只生成配置和验证 Compose，不拉取大镜像、不执行构建、不启动容器，也不改变已有 cutedsl_dev。

## 1. 镜像选择

官方镜像 **`vllm/vllm-openai:v0.33.0` 已存在**，Docker Hub 标签 API 已验证 linux/amd64 与 linux/arm64。0.33.0 官方发布说明指定 CUDA 13.0、PyTorch 2.14.0。当前主机未发现该官方镜像的本地标签，需要后续下载。

- 多架构 digest：`sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`
- amd64 digest：`sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3`
- amd64 压缩层合计约 9.74 GB；拉取解压和构建还需要额外磁盘空间。
- 查询原始信息保存在 `vllm033-image-metadata.json`。

已有官方运行时，因此不从源码编译 vLLM。提供的 `Dockerfile.vllm033_dev` 仅在固定 digest 的官方镜像上安装 sudo、Git 等基础工具，创建普通用户及免密 sudo，保留官方 Python/CUDA/PyTorch 包组合，并将默认 API server 入口改为 `sleep infinity`，方便手动开发。

这不是现有 `ai-system/cutedsl-dev:ncu-preserved` 镜像的完整复制；原镜像手动加入的 Nsight Compute 等工具不保证存在。新镜像的实际 nvcc、CuTe DSL、Python 版本在构建后按下文验证。不要把旧 `/workspace/venvs/vllm-0112` 加入新容器 PATH。

## 2. 配置与现有容器的对应关系

| 项目 | 新配置 |
|---|---|
| Compose 文件 | `container/docker-compose-vllm033-dev.yaml` |
| Dockerfile | `dockerfile/Dockerfile.vllm033_dev` |
| 参数模板 | `container/.env.vllm033.example` |
| Compose project | `vllm033-dev`，独立于 cutedsl 项目 |
| 镜像 | `ai-system/vllm-dev:0.33.0-cu130` |
| 容器/service | `vllm033_dev` |
| 普通用户 | `vllmdev`，UID/GID 1000:1000，与 cdky/旧容器一致 |
| sudo | `NOPASSWD:ALL`，可用 `sudo -n true` 验证 |
| GPU | NVIDIA，count: all，capabilities: [gpu] |
| 容器权限 | 参照旧配置保留 privileged: true，另列 SYS_ADMIN |
| 工作目录 | `/workspace` |
| 端口范围 | 宿主机和容器 TCP 7501–7599，同号映射 |
| 共享内存 | 私有 IPC，默认 16gb |
| bind mount | `/home/cdky/vic/project/02_kycv/WVisonGuard/train_part` → `/workspace` |
| 网络 | 独立 Compose bridge：`vllm033-dev_default`（默认项目名） |
| 子网/网关 | Docker 从地址池自动分配，无需手动指定 |
| 容器 IP | 在新项目网络中自动分配 |

新项目使用独立 bridge 网络，不依赖 cutedsl-dev_default。Compose up 创建网络，down 删除本项目网络；不影响旧 cutedsl 网络。192.168.88.91 是物理机局域网 IP，Docker bridge 的子网和网关是容器内部网络地址。客户端通过 192.168.88.91:7501 进入宿主机，再由端口映射转发到新容器的 7501。此前记录的 172.22.0.0/16 和 172.22.0.1 属于旧容器网络，不需要写入新配置。

**共享内存差异：**旧文件同时设置 `ipc: host` 和 `shm_size`，但 host IPC 使用宿主机 `/dev/shm`。新配置使用 `ipc: private` 使 16gb 设置生效。若确实需要跨容器共享 host IPC，可改为 host，但应接受 shm_size 不再控制其大小。

新容器没有重建 0.14.1 虚拟环境，也没有自动启动模型。两容器共享 GPU 和可写工作目录，不代表显存或文件相互隔离；构建 Dockerfile 不会递归修改挂载目录权限。

## 3. 把配置放到 Linux 主机

推荐复制整个 docker 目录到主机新的独立目录，避免覆盖已有 cutedsl 配置。Windows PowerShell：

```powershell
ssh cdky@192.168.88.91 'mkdir -p /home/cdky/vllm033-deployment'
scp -r D:\workspace\learing\AI_system\docker cdky@192.168.88.91:/home/cdky/vllm033-deployment/
ssh cdky@192.168.88.91
```

下面全部在 **Linux 宿主机** 执行，不是在旧 cutedsl 容器内执行：

```bash
cd /home/cdky/vllm033-deployment/docker/container
# 首次执行；已经存在时保留自己的参数文件。
test -e .env.vllm033 || cp .env.vllm033.example .env.vllm033

id -u
id -g
nvidia-smi
docker compose version
test -d /home/cdky/vic/project/02_kycv/WVisonGuard/train_part
```

实测 Compose 是 v2.21.0；宿主机 NVIDIA Container Toolkit 已能支持现有 GPU 容器。编辑 `.env.vllm033`，核对工作目录和用户编号；无需设置 CUTEDSL_NETWORK，旧参数文件里该变量可以删除。路径指向 Linux 宿主机，不能填写 Windows 的 D:\ 路径。`create_host_path: false` 使路径错误时直接报错，不悄悄创建空目录。

如果复制到了其它 Linux 目录，只需修改 cd；build context 相对于 Compose 文件解析，仍为 `../dockerfile`，挂载目录则由 `.env.vllm033` 的绝对路径决定。

## 4. 检查、构建与启动

每条命令都显式指定新的参数文件与 Compose 文件，不读取旧 cutedsl 的项目参数来替换容器。

```bash
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml config

# 显示已监听端口，人工确认 7501–7599 没有冲突；不结束任何现有进程。
ss -ltn

# 首次构建会下载固定的官方基础镜像及 apt 包。
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml build --pull

docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml up -d --no-build

docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml ps

docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml exec vllm033_dev bash
```

没有镜像下载权限或网络故障时，build 会失败；不要在旧容器里临时 pip upgrade 来替代。固定 digest 保证基础镜像内容固定，apt 仓库包并未锁定到快照，所以整个派生镜像不是逐字节可复现构建。

已通过远端 Docker Compose v2.21.0 的 config 解析检查：GPU count=all、99 个端口映射、16 GiB shm、独立默认网络及 bind mount 均正确解析。检查通过标准输入完成，build context 的实际路径仍需在复制后的目录用上述 config 命令确认。本次没有实际构建镜像，构建成功和以下验收都需后续执行确认。

## 5. 容器内检查

```bash
id
sudo -n true
sudo -n id
pwd
df -h /dev/shm
nvidia-smi
command -v python3
command -v vllm
command -v nvcc && nvcc --version
```

预期：普通用户 uid/gid 与 `.env.vllm033` 一致、sudo 输出 uid=0、工作目录 /workspace、/dev/shm 大小 16G、可以看到 GPU。

```bash
python3 - <<'PY'
import sys
import importlib.metadata as m
import torch
print('Python:', sys.version)
print('vLLM:', m.version('vllm'))
print('Torch:', torch.__version__)
print('Torch CUDA:', torch.version.cuda)
assert m.version('vllm') == '0.33.0'
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
x = torch.ones((128, 128), device='cuda')
assert (x @ x)[0, 0].item() == 128
print('Basic CUDA computation OK')
try:
    print('CuTe DSL:', m.version('nvidia-cutlass-dsl'))
except m.PackageNotFoundError:
    print('CuTe DSL package not found')
PY

vllm serve --help
```

基础矩阵运算成功不等于模型 backend/量化/图编译全部验证通过。首次模型运行仍可能编译 JIT kernels。不要从宿主机挂载 CUDA 12.9 到新容器 `/usr/local/cuda`，也不要套用旧镜像的 CUDA/Python 环境变量。

## 6. 手动启动 Qwen3-8B API

在容器内先检查实际模型目录。以下使用明确占位符，必须替换成真实绝对路径：

```bash
MODEL=/workspace/weights_pub/替换为实际Qwen3-8B目录
test -f "$MODEL/config.json" || { echo '请先设置 MODEL 的真实目录'; exit 1; }

vllm serve "$MODEL" \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 --port 7501 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

新容器常驻 shell，并不会自动常驻上述模型服务；终端前台运行可用 Ctrl+C 停止该模型进程。需要后台守护服务时应另设服务 command，不把这次开发 Compose 的启动成功当作模型就绪。

现有 GPU 上如果已有其它大模型，新服务可能显存不足。`gpu-memory-utilization` 不会为不同容器隔离显存；先核对 `nvidia-smi` 再选择模型、上下文和并发，不自动停止旧服务。

原项目 `capstone/runtime_training/scripts/serve.py` 强制要求 0.14.1，因此新版先用官方 `vllm serve` 命令，不修改原 W31 启动器和 requirements。

Windows 验证：

```powershell
Invoke-RestMethod http://192.168.88.91:7501/health
Invoke-RestMethod http://192.168.88.91:7501/v1/models
$body = @{
    model = 'qwen3-8b'
    messages = @(@{role='user'; content='用一句话解释KV cache。'})
    max_tokens = 32
    temperature = 0
    chat_template_kwargs = @{enable_thinking=$false}
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://192.168.88.91:7501/v1/chat/completions -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

使用宿主机 `192.168.88.91:7501` 访问，容器 bridge IP 仅供 Docker 网络内通信。容器服务要监听 0.0.0.0；监听 127.0.0.1 不会被端口映射正常转发。宿主机防火墙需允许所用端口。8000/8888 不在这次映射范围，Jupyter 如需使用应另安装并选择 7501–7599 内空闲端口。

## 7. 管理与故障定位

从 Linux 宿主机的 container 目录执行：

```bash
# 查看常驻容器日志；手动 exec 启动的前台模型日志通常在其终端。
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml logs --tail=100

# 停止/重新启动此项目。
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml stop
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml start

# 删除此项目容器和它的独立网络，不删除 bind mount 的工作文件。
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml down
```

- 网络地址池耗尽或与局域网/VPN 路由冲突：先检查 docker network ls 和宿主机路由，再选择不冲突的地址池；默认不硬编码子网。
- 端口被占用：先识别占用者，调整自己的端口映射或服务端口，不结束未知进程。
- bind mount 写入失败：确认主机目录所有者与 UID/GID；不递归 chown 整个共享工作目录。
- sudo 失败：确认运行的是新派生镜像，重新构建；官方基础镜像不保证普通用户免密 sudo。
- GPU 不可见：检查宿主机驱动、NVIDIA Container Toolkit 和 Compose 解析后的 devices 预留。
- import/模型失败：保存镜像 digest、日志、torch.version.cuda 和实际 backend；不要混装旧环境包。

## 8. 来源

- [官方 0.33.0 发布说明及镜像表](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)
- [Docker Hub 标签 API](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.33.0)
- [官方 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile)
- [Docker Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA CUDA 驱动兼容](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

GitHub 的该发布展示名称为 v0.33.0，实际 tag 为 v0.33.0rc3；Docker Hub v0.33.0 标签已单独查询确认。本配置使用明确镜像 digest，不依赖 GitHub tag 推导镜像是否存在。

## 从旧版共享网络配置切换

若此前已启动新容器，请先同步更新后的 Compose 文件，然后在 Linux 的 container 目录执行以下命令，仅重建新服务，不操作 cutedsl_dev：

```bash
docker compose --env-file .env.vllm033 -f docker-compose-vllm033-dev.yaml up -d --no-build --force-recreate vllm033_dev
```

这会中断新容器内正在运行的进程；手动启动的模型需重新启动。若尚未构建/启动，则直接按第 4 节首次执行即可。
