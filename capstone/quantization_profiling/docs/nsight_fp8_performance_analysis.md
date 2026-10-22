# Qwen3-8B：FP8 / BF16 / AWQ 的 Nsight 性能分析实操

适用目录：`capstone/quantization_profiling`。更新日期：2026-10-22。

主分析对象：`fp8-local`（下文也称 fp8local）、`fp8-official`（用户所称 fp8origin）；对照：`bf16`、`awq`。用户提到的 bf6 按本实验的 BF16 理解。

本文针对 Linux GPU 目标机，按当前项目的 **vLLM 0.14.1、Qwen3-8B、RTX 4090D、TP=1** 设计。GPU 和版本必须由原实验 `environment.json`、`run.json` 最终确认。Windows 工作区用于阅读、编辑与打开报告；不要把 Linux 命令粘进 PowerShell。本文未在远端 GPU 上实际采集，所有 kernel 性能归因仍需下述证据验证。

## 0. 阅读与执行顺序

1. 第 1～3 节：理解现有数据与优先假设。
2. 第 4～6 节：保存环境、核对模型身份、定位实际后端。
3. 第 7～9 节：Nsight Systems 对四个 HTTP 服务分别抓一份时间线，分清 GPU 工作和 host 空洞。
4. 第 10～13 节：用离线入口抓 prefill / decode 的代表热点，再用 Nsight Compute 看硬件指标。
5. 第 14～17 节：映射到模型层、做受控对照、整理证据与排障。

最小第一轮：四个模型各抓 `w1024_o128` 的 1 个正式请求；FP8 两组再抓 `w4096_o256`；不要一开始把 80 个请求交给 NCU。

## 1. 现有结果究竟说明了什么

实验：`20261022T023823Z_8746d6b6`；快照 SHA256：`5588398462ab77d0d0360a698300463a6224f2798b04bc9cb120f2429c2f68a2`；有效请求子集 SHA256：`05f164183a6dd10561d5c213eef38e81eebe63f1edf9b5701748e08bbc3fc99d`。

有效协议为每模型、每负载 1 轮，5 次预热，80 次正式请求，并发 1。全部 20 组通过，正式请求总计 1600 次。通过指协议与请求成功，不表示量化一定加速，也不代表质量已通过。

### 1.1 相对吞吐：按同负载比较，不跨精度合并

| 负载 | 官方 FP8 / BF16 | 本地 FP8 / BF16 | 本地 FP8 相对官方 FP8 | AWQ / BF16 |
|---|---:|---:|---:|---:|
| w128_o128 | 56.11% | 55.29% | -1.47% | 98.86% |
| w1024_o128 | 57.95% | 58.58% | +1.08% | 100.35% |
| w4096_o256 | 61.77% | 61.34% | -0.70% | 102.97% |
| w8192_o128 | 72.89% | 73.65% | +1.04% | 100.84% |
| w1024_o512 | 56.25% | 56.37% | +0.22% | 100.88% |

计算：`相对吞吐 = FP8 token/s / BF16 token/s`；`本地相对官方变化 = local / official - 1`。

**直接观察：** 两份 FP8 的吞吐都明显低于 BF16，而 local 与 official 吞吐差异只在约 -1.5%～+1.1%。目前更值得先查它们共同的运行路径。不能用一轮结果认定两种模型性能等价，也不能把某组 p95 差异当作稳定收益。

### 1.2 首字更快，总体更慢：优先调查 decode

例如 `w4096_o256`：

| 模型 | TTFT 文本事件 p95 | E2E p95 |
|---|---:|---:|
| BF16 | 0.464 s | 5.836 s |
| 官方 FP8 | 0.337 s | 8.634 s |
| 本地 FP8 | 0.337 s | 8.818 s |

`w8192_o128` 的 FP8 首字也更快。长输入 prefill 可能受益，但后续生成开销足以抵消收益。这是**待验证的阶段归因假设**，还不是 GPU kernel 结论。

注意三点：

- `E2E p95 - TTFT p95` 不是 decode p95；两个分位数可能对应不同请求。
- `(E2E - TTFT)/(输出 token 数-1)` 只有按同一请求计算才有意义；这里仍是客户端文本事件近似。
- 本项目吞吐按正式阶段请求起止墙钟范围计算，包含 prefill、调度、网络等。`1 / token/s` 不能直接解释为纯 GPU TPOT。

### 1.3 当前还不能回答的事

没有原实验逐请求记录、GPU 时间线和计数器，就不能断言：FP8 完全回退成 BF16、未使用 Tensor Core、是本地量化错误、是 CPU 瓶颈，或者是功耗限制。后续步骤分别排除这些情况。

## 2. 为什么这套 FP8 很可能更慢

### 2.1 先看项目自己的设置

| 项目 | 当前配置 / 代码行为 | 对分析的影响 |
|---|---|---|
| 官方 FP8 | 原生 `fp8`，E4M3，128×128 分块，动态 activation | 不等于普通 per-tensor FP8 |
| 本地 FP8 | `FP8_BLOCK`，通常 `compressed-tensors` 自动识别 | 应核对真实 config，不能只看目录名 |
| 两组 FP8 的 dtype | `bfloat16` | 非量化部分和输出 dtype；不表示 FP8 权重被取消 |
| AWQ | `awq_marlin` + BF16 | 与原始 AWQ kernel 不同 |
| 并发 / max_num_seqs | 1 / 1 | decode 的 token 行数通常 M=1，效率不同于大批量 |
| enforce_eager | 四组均为 true | 关闭 CUDA Graph；小 kernel 和 CPU launch 开销更突出 |
| max_num_batched_tokens | 2048 | 4096/8192 输入可能被 chunk，不能当一次大 prefill |
| prefix caching | 关闭 | 重复输入不应跳过 prefill |
| KV cache dtype | auto | 当前 BF16 激活下通常为 BF16 KV；不自动随权重量化变 FP8 |
| max_model_len | 每 case 输入+输出 | profiling 首轮要保留相同上限 |

依据：`configs/*.yaml`、`scripts/model_profiles.py`、`scripts/run_model.py`、`scripts/benchmark.py`。配置文件是 JSON 内容，扩展名为 `.yaml`。

### 2.2 4090D 支持 FP8，不代表支持这条最优分块路径

本仓库的[固定版本调用路径分析](vllm_0_14_1_fp8_linear_callgraph.md)给出：在 vLLM 0.14.1 的常规 CUDA 分支，Ada SM89 的这类 BLOCK FP8 不满足 DeepGEMM、FlashInfer SM90 和 CUTLASS BLOCK 条件，预期走 Triton block GEMM；动态激活量化本身仍可能走 CUDA kernel。

这个结论是对特定版本、特定 scheme 的源码推断。必须用第 6 节打印目标机安装源码，并用 nsys 观察 kernel 交叉验证。仓库已有分析文档引用的 `reference/` 缓存当前未随本次检出提供，因此本文不假装读取了那些不存在的缓存文件。

预期关系：

```text
官方 fp8 loader ───────────┐
                         ├─ W8A8BlockFp8LinearOp
本地 compressed-tensors ──┘     ├─ 动态 group128 激活量化
                               ├─ block FP8 GEMM（SM89 常规分支预期 Triton）
                               └─ BF16 输出 + 其它模型算子
```

两组 loader 不同，但可能汇入同一条主要计算路径；这与吞吐接近相符，仍需验证实际 dispatch。不要强行给 compressed-tensors checkpoint 加 `--quantization fp8` 来“统一后端”，这会改变 loader 语义或报错。

**显式 Marlin 是独立对照路线。** benchmark 现支持 `--model fp8-origin --fp8-backend marlin --enforce-eager false`。vLLM 0.14.1 的原生 FP8 loader 通过 `VLLM_TEST_FORCE_FP8_MARLIN=1` 进入 W8A16；本地 compressed-tensors FP8_BLOCK/W8A8 不响应这个变量，不能据此把它标记为 Marlin。auto 模式显式将该变量设为 0，以免继承终端残留。完整命令、限制和报告字段见 [benchmark 执行开关](benchmark_backends.md)。原表的 W8A8 性能与 Marlin 的 W8A16 性能应分别记录。

### 2.3 M=1 的 decode 与长 prefill 是两种问题

线性层可写为 `C[M,N] = A[M,K] × W[N,K]^T`。

- 单请求 decode：通常 M=1；权重读取、launch、scale/quant、tile 未充分利用可能占主导。
- prefill：M 为该次 scheduler step 的 token 行数；可能约 128/1024/2048，而不是总输入长度。
- FP8 多了 activation quant 和 scale 处理；权重读取减少的好处不保证盖过这些开销。
- Triton block GEMM 可以使用原生 FP8 指令；“Triton”不等于“软件模拟”。应看 NCU SASS 和 Tensor Core 指标。
- AWQ 权重更小，但反量化、融合实现、非线性层、调度等仍有成本；其与 BF16 接近也不能单靠权重大小解释。

第一优先级假设：**SM89 的 BLOCK FP8 后端 + 小 M decode + eager 的共同开销**。备选假设：GPU 时钟/其它进程、动态量化、额外 cast/copy、scheduler/IPC、配置或模型身份不一致。

## 3. 工具分工与证据标准

| 工具 | 要回答的问题 | 不能拿来替代什么 |
|---|---|---|
| 无 profiler 的 benchmark | 真正的吞吐、p50/p95、失败率 | 不提供 kernel 原因 |
| Nsight Systems / nsys | prefill/decode 时间线、launch 数、GPU gaps、同步/拷贝 | 不单凭 kernel 名称确认硬件指令 |
| Nsight Compute / ncu | 热点 kernel 的耗时、带宽、Tensor Core、occupancy、SASS | 不把 replay 下 HTTP E2E 当生产性能 |
| PyTorch profiler | 模型 op、shape、调用栈到 CUDA kernel 的映射 | 不与 nsys/ncu 同时启动 GPU profiler |
| nvidia-smi | GPU UUID、其它进程、时钟/功耗/温度、设备级显存 | 采样 utilization 不等于 Tensor Core 利用率 |

理想因果链：同一请求 → 同一阶段 → 同一线性层/shape → 实际 kernel → 关键计数器 → 单变量修改 → 无 profiler 复测。

## 4. Step 1：准备目录、环境和工具

下文所有 Bash 命令在 GPU 目标机执行；实际路径不同只改本节变量。每个新终端都重新设置相同变量。

```bash
cd /workspace/AI_system
source /workspace/venvs/vllm-0112/bin/activate
export REPO="$PWD"
export DEMO="$REPO/capstone/quantization_profiling"
export MODEL_ROOT=/workspace/weights_pub
export EXP="$DEMO/log/benchmark_20261022T023823Z_8746d6b6"
export OUT="$REPO/out/quantization_profiling/nsight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT" "$OUT/nsys" "$OUT/ncu" "$OUT/torch" "$OUT/evidence"
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export PYTHONPATH="$DEMO/scripts${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' "$OUT"
```

目的：用独立目录保存诊断产物；不修改冻结输入、原报告或原协议。`CUDA_VISIBLE_DEVICES=0` 是示例，须核对 GPU UUID；工具中的逻辑 device 0 与宿主机编号不总是一致。

### 4.1 保存环境与 GPU 状态

```bash
python "$DEMO/scripts/collect_environment.py" \
  --expected-gpu 4090D --output "$OUT/environment.json"
python -m pip freeze > "$OUT/pip-freeze.txt"
python -m pip check > "$OUT/pip-check.txt" 2>&1
nvidia-smi -L > "$OUT/gpus.txt"
nvidia-smi -q > "$OUT/nvidia-smi-q.before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv > "$OUT/processes.before.csv"
python - <<'PY'
import torch, vllm
print('torch=', torch.__version__, 'cuda_build=', torch.version.cuda)
print('vllm=', vllm.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, 'CC=', (p.major, p.minor), 'SMs=', p.multi_processor_count)
PY
```

目的：确认是否真为 4090D / CC 8.9 / vLLM 0.14.1；检查其它占卡进程与依赖混装。这里单独的探针 Python 会退出，不在启动 vLLM worker 的父进程提前初始化 CUDA。

另开一个终端，在采集期间记录设备状态：

```bash
nvidia-smi --query-gpu=timestamp,uuid,pstate,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv -lms 500 > "$OUT/telemetry.csv"
```

结束后在这个监控终端按 Ctrl+C。字段不支持时先查 `nvidia-smi --help-query-gpu` 再删不支持项。不要因为 FP8 读写更少就预设所有模型时钟相同。

### 4.2 检查工具能力，保存实际 help

```bash
command -v python nsys ncu
nsys --version > "$OUT/nsys-version.txt"
ncu --version > "$OUT/ncu-version.txt"
nsys profile --help > "$OUT/nsys-profile-help.txt"
nsys stats --help > "$OUT/nsys-stats-help.txt"
nsys stats --help-reports > "$OUT/nsys-reports.txt"
ncu --help > "$OUT/ncu-help.txt"
ncu --list-sets > "$OUT/ncu-sets.txt"
ncu --list-sections > "$OUT/ncu-sections.txt"
ncu --query-metrics > "$OUT/ncu-metrics.txt"
nsys status --environment
```

目的：本文不能假定你已安装某个最新 Nsight；参数、report、section、metric 必须以目标机工具为准。`nsys status --environment` 的 CPU sampling 失败不一定阻止 CUDA tracing。

未安装时先检查目标发行版与已有 CUDA 源：

```bash
cat /etc/os-release
uname -m
apt-cache search nsight-systems
apt-cache search nsight-compute
```

在已配置 NVIDIA 软件源的 Ubuntu 上，Systems 的 CLI 包通常为 `nsight-systems-cli`，Compute 常见为带版本号的 `nsight-compute-*`。按 `apt-cache` 实际输出选择包，或使用 [Nsight Systems 下载](https://developer.nvidia.com/nsight-systems/get-started)、[Nsight Compute 下载](https://developer.nvidia.com/nsight-compute/get-started)的独立安装包。安装 CLI 不必为此升级实验的 torch/vLLM/CUDA 驱动。目标是工具支持当前驱动与 GPU，GUI 能打开该版本报告。

### 4.3 GPU 计数器权限与容器

先只读检查：

```bash
id
cat /proc/driver/nvidia/params | grep -i Profiling
cat /proc/sys/kernel/perf_event_paranoid
```

遇到 `ERR_NVGPUCTRPERM`，按 [NVIDIA 官方权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)处理。最小可行方法通常是让管理员为本次 NCU 运行授予所需权限；使用 sudo 时显式使用当前虚拟环境的 Python 和 `ncu` 绝对路径，避免误入系统 Python。不要为了 GPU trace 无条件关闭整机权限限制或卸载在用驱动。

Docker 中，**在运行模型的容器里面执行 nsys/ncu**。容器应能看到同一个 GPU、权重、repo 和输出挂载；NCU 计数器可能需要启动时增加 `--cap-add=SYS_ADMIN`，CPU sampling 可能另外需要 perf/ptrace 权限。仅 CUDA tracing 首轮用 `--sample=none --cpuctxsw=none` 降低权限需求。不要只在宿主机包一层 `nsys docker exec ...`，也不要在 HTTP client 外套 profiler 后期待看到服务 GPU。

容器权限规则与驱动版本相关，参考上面的官方说明；`root` 身份本身不保证容器已获得 host 计数器权限。

## 5. Step 2：核对原实验与 checkpoint 身份

### 5.1 先找到真实产物，别把工作区配置当成当时配置

```bash
find "$DEMO/log" -maxdepth 2 -name run.json -print
ls -l "$EXP/run.json" "$EXP/environment.json" "$EXP/selected_requests.jsonl"
python -m json.tool "$EXP/run.json" > "$OUT/evidence/original-run.pretty.json"
find "$EXP" -maxdepth 3 -type f \
  \( -name server.log -o -name summary.json -o -name gpu.jsonl \) -print
```

目的：确认原实验 GPU、版本、实际 command、enforce_eager、量化参数与模型身份。当前 Windows 检出未包含这个实验的完整原始目录；应在原 GPU 目标机找。没有文件时，保留“仅由用户表格推断”的边界。

```bash
if [[ -f "$EXP/selected_requests.jsonl" ]]; then
  export REQUESTS="$EXP/selected_requests.jsonl"
else
  export REQUESTS="$DEMO/data/frozen/requests.jsonl"
fi
sha256sum "$REQUESTS" "$DEMO/data/frozen/manifest.json" \
  > "$OUT/evidence/input-file-hashes.txt"
```

`selected_requests_sha256` 是代码对请求对象做 canonical JSON 后的逻辑摘要；上面的 `sha256sum` 是文件字节摘要，两者不能直接比较相等。

### 5.2 逐一读取四个 config 与量化元数据

```bash
python - <<'PY' > "$OUT/evidence/checkpoint-configs.txt"
import json, os
from pathlib import Path
root = Path(os.environ['MODEL_ROOT'])
for name in ('Qwen3-8B', 'Qwen3-8B-AWQ', 'Qwen3-8B-FP8', 'Qwen3-8B-FP8-local'):
    p = root / name / 'config.json'
    c = json.loads(p.read_text())
    print('\nMODEL:', p.parent)
    for k in ('model_type', 'architectures', 'dtype', 'torch_dtype',
              'hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'head_dim'):
        print(k, c.get(k))
    print(json.dumps(c.get('quantization_config'), ensure_ascii=False, indent=2))
PY
```

核对：

| 模型 | 主要检查 |
|---|---|
| BF16 | 无权重量化配置，实际加载 BF16；架构与基础模型匹配 |
| AWQ | 4 bit、group size、zero point；服务实际 `awq_marlin` |
| 官方 FP8 | `quant_method=fp8`、E4M3、`activation_scheme=dynamic`、`weight_block_size=[128,128]` |
| 本地 FP8 | compressed-tensors 的 weights float/8bit/block、input_activations float/8bit/dynamic/group128；忽略层等 |

另保存 tokenizer/chat template 指纹、量化转换 manifest、源 revision。两组 FP8 的 scale 数值不同本来就可能合理；要核对粒度、方向、shape 与 loader 约定。不能仅比较配置文本就证明权重同源或数值等价。

必要时在空闲阶段对权重做一次全量 hash，别放进计时窗口：

```bash
find "$MODEL_ROOT/Qwen3-8B-FP8-local" -maxdepth 1 -name '*.safetensors' \
  -exec sha256sum {} \; > "$OUT/evidence/local-weight-hashes.txt"
find "$MODEL_ROOT/Qwen3-8B-FP8" -maxdepth 1 -name '*.safetensors' \
  -exec sha256sum {} \; > "$OUT/evidence/official-weight-hashes.txt"
```

### 5.3 不加载大权重，查看真实存储 dtype / shape

```bash
python - <<'PY' > "$OUT/evidence/tensor-headers.txt"
import os
from collections import Counter
from pathlib import Path
from safetensors import safe_open
for name in ('Qwen3-8B', 'Qwen3-8B-AWQ', 'Qwen3-8B-FP8', 'Qwen3-8B-FP8-local'):
    counts, shown = Counter(), 0
    print('\nMODEL', name)
    for file in sorted((Path(os.environ['MODEL_ROOT']) / name).glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as f:
            for key in f.keys():
                view = f.get_slice(key)
                counts[str(view.get_dtype())] += 1
                if shown < 16 and ('weight' in key or 'scale' in key):
                    print(key, view.get_shape(), view.get_dtype())
                    shown += 1
    print('tensor dtype counts:', dict(counts))
PY
```

目的：确认是量化权重文件而非只改 config；此处 dtype 计数按 tensor 个数，不是参数数/字节数。读取 header 不等于验证运行时 GEMM 指令。

## 6. Step 3：查目标机实际安装源码与 backend 条件

```bash
python - <<'PY' > "$OUT/evidence/fp8-dispatch-source.txt"
import inspect
import vllm.model_executor.layers.quantization.utils.fp8_utils as b
import vllm.model_executor.layers.quantization.fp8 as native
import vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 as ct
from vllm.platforms import current_platform
print('installed fp8_utils:', b.__file__)
print('capability:', current_platform.get_device_capability())
print('\nNATIVE LOADER\n', inspect.getsource(native.Fp8LinearMethod))
print('\nCOMPRESSED TENSORS\n', inspect.getsource(ct.CompressedTensorsW8A8Fp8))
print('\nBLOCK DISPATCH\n', inspect.getsource(b.W8A8BlockFp8LinearOp))
PY
```

若某个符号在实际安装版本改名，先打印 `__file__`、`dir(module)`，按该版本查；不要修改生产库让文档命令勉强成立。

```bash
VLLM_DIR=$(python -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent)')
rg -n 'cutlass_block_fp8_supported|is_deep_gemm_supported|_run_triton|_run_cutlass|_run_deepgemm|BLOCKSCALE_FP8' \
  "$VLLM_DIR/model_executor/layers/quantization" "$VLLM_DIR/utils" \
  > "$OUT/evidence/backend-predicates.txt"
python - <<'PY' > "$OUT/evidence/runtime-env.txt"
import os
# Only collect execution controls; do not dump API keys or other credentials.
for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
             'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_USE_V1', 'VLLM_USE_V2_MODEL_RUNNER',
             'VLLM_USE_DEEP_GEMM', 'VLLM_USE_DEEP_GEMM_E8M0',
             'VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER', 'VLLM_BATCH_INVARIANT',
             'VLLM_ATTENTION_BACKEND', 'VLLM_USE_TRITON_FLASH_ATTN',
             'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_MODULE_LOADING'):
    print(name, repr(os.environ.get(name)))
PY
```

没有 `rg` 时使用 `grep -R -n -E`。目的：分清普通 FP8 CUTLASS 支持与 BLOCK FP8 支持；发现实际环境变量、源码补丁和版本差异。

启动日志只能证明 dtype、loader 等初始化信息；即使出现 CUTLASS/FlashInfer 名称，也不保证每个 shape 都走它。后续以 CUDA kernel 和 operator 对照为准。[v0.14.1 分块 FP8 源码](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/layers/quantization/utils/fp8_utils.py)。

## 7. Step 4：为 HTTP 服务准备可重复采集入口

本节生成两个小工具到 `$OUT`，不会改项目脚本。复用原 `/v1/completions`、固定 token IDs 和 `benchmark.perform` 校验，避免把自然语言 smoke test 当成原 workload。

### 7.1 生成四模型通用服务启动器

```bash
cat > "$OUT/launch_server.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${DEMO:?}" "${MODEL_ROOT:?}" "${PROFILE:?}" "${CASE:?}"
case "$CASE" in
  w128_o128) MAXLEN=256 ;;
  w1024_o128) MAXLEN=1152 ;;
  w4096_o256) MAXLEN=4352 ;;
  w8192_o128) MAXLEN=8320 ;;
  w1024_o512) MAXLEN=1536 ;;
  *) echo "unknown CASE: $CASE" >&2; exit 2 ;;
esac
Q=()
case "$PROFILE" in
  bf16) MODEL="$MODEL_ROOT/Qwen3-8B" ;;
  awq) MODEL="$MODEL_ROOT/Qwen3-8B-AWQ"; Q=(--quantization awq_marlin) ;;
  fp8-official) MODEL="$MODEL_ROOT/Qwen3-8B-FP8"; Q=(--quantization fp8) ;;
  fp8-local) MODEL="$MODEL_ROOT/Qwen3-8B-FP8-local" ;;
  *) echo "unknown PROFILE: $PROFILE" >&2; exit 2 ;;
esac
E=()
[[ ${EAGER:-1} == 1 ]] && E=(--enforce-eager)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export PYTHONPATH="$DEMO/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tokenizer "$MODEL" --dtype bfloat16 "${Q[@]}" \
  --host 127.0.0.1 --port "${PORT:-7081}" --served-model-name local-model \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.75 \
  --max-num-seqs "${MAX_SEQS:-1}" --tensor-parallel-size 1 \
  --max-num-batched-tokens "${BATCH_TOKENS:-2048}" \
  --seed 42 --generation-config vllm \
  --no-enable-prefix-caching --enable-chunked-prefill "${E[@]}" \
  --middleware profiling_middleware.ProfilingMiddleware \
  --enable-force-include-usage \
  --profiler-config '{"profiler":"cuda"}'
SH
```

目的：保持原服务的关键参数，只额外设置 CUDA profiler 控制。原 `run_model.py` 不暴露 `--profiler-config`，所以这里显式构造相同的底层 API server 命令，不给它传不存在的参数。

首轮 `EAGER=1` 保留原实验行为。改成 `EAGER=0` 是独立消融，不属于原结果。`max_model_len` 包括原始输入+输出，不统一改为 8192。

vLLM 0.14.1 支持 `--profiler-config '{"profiler":"cuda"}'`，等价的细分形式见官方帮助。先在目标环境确认：

```bash
python -m vllm.entrypoints.openai.api_server --help > "$OUT/vllm-server-help.txt"
grep -n -A 8 'profiler-config' "$OUT/vllm-server-help.txt"
```

依据：[v0.14.1 profiler 配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/config/profiler.py)。不是所有旧版 vLLM 都接受这个选项。

### 7.2 生成冻结请求回放工具

```bash
cat > "$OUT/replay_http.py" <<'PY'
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from benchmark import perform
from test_server import Client


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests-file', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--base-url', default='http://127.0.0.1:7081')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control', action='store_true')
    a = p.parse_args()
    if a.count < 1:
        p.error('--count must be positive')
    rows = [json.loads(s) for s in a.requests_file.read_text().splitlines() if s.strip()]
    def pick(stage, n):
        selected = sorted((r for r in rows if r['case'] == a.case
                           and r['phase'] == stage), key=lambda r: r['id'])[:n]
        if len(selected) != n:
            raise ValueError(f'need {n} {stage} rows for {a.case}')
        return selected
    warm, measured = pick('warmup', 5), pick('measure', a.count)
    cargs = SimpleNamespace(base_url=a.base_url, timeout=a.timeout,
                            api_key_env='VLLM_API_KEY', no_proxy=True)
    client = Client(cargs)
    client.health()
    config = {'served_model_name': 'local-model', 'profile': a.profile}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    def run(items, stage, f):
        for item in items:
            entry = perform(cargs, config, item, 1, stage)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()
            print(stage, item['id'], entry['ok'], flush=True)
            if not entry['ok']:
                raise RuntimeError(entry)
    def control(path):
        with client.open(path, {}) as response:
            response.read()
            print(path, response.status, flush=True)
    with a.output.open('x', encoding='utf-8') as f:
        run(warm, 'warmup', f)
        if a.control:
            control('/start_profile')
        try:
            run(measured, 'measure', f)
        finally:
            if a.control:
                control('/stop_profile')


if __name__ == '__main__':
    main()
PY
```

此工具复用项目校验：实际 input/output token 数、固定输出长度、服务器 profiling 扩展。`--control` 在预热后调用 worker profiler，再回放正式请求，最后关闭；没有 `--control` 可做不触发 profiler 的连通性验证。这里的请求数量仅用于诊断，不是正式 80 请求协议验收。

### 7.3 单独确认 profiler API 存在

服务启动后先检查路由：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:7081/openapi.json \
  > "$OUT/evidence/openapi.json"
python - <<'PY'
import json, os
from pathlib import Path
paths = json.loads((Path(os.environ['OUT'])/'evidence/openapi.json').read_text())['paths']
print([s for s in paths if 'profile' in s])
PY
```

必须看到 `/start_profile` 与 `/stop_profile`。若 404，先核对实际版本、profiler 配置、服务路由与部署权限，不能把 HTTP 404 当作采集成功。是否需要开发模式取决于安装版本；不默认添加 `VLLM_SERVER_DEV_MODE=1`。本流程绑 loopback，用本机受控诊断服务。

API 与 worker 内部 CUDA profiler 的联系见 [v0.14.1 worker profiler 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/profiler/wrapper.py)：控制调用最终需要在持有 CUDA context 的 worker 里执行，HTTP client 上的 NVTX 不能替代它。

## 8. Step 5：Nsight Systems 抓 HTTP 服务时间线

### 8.1 终端 A：以 local FP8、1024→128 为例

先保存跨终端变量：

```bash
declare -p REPO DEMO MODEL_ROOT EXP OUT REQUESTS CUDA_VISIBLE_DEVICES \
  VLLM_WORKER_MULTIPROC_METHOD HF_HUB_OFFLINE PYTHONPATH > "$OUT/session.env"
export PROFILE=fp8-local
export CASE=w1024_o128
export EAGER=1
export PORT=7081
set -o pipefail
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --kill=none \
  -o "$OUT/nsys/${PROFILE}_${CASE}_eager" \
  bash "$OUT/launch_server.sh" \
  2>&1 | tee "$OUT/nsys/${PROFILE}_${CASE}_server.log"
```

参数目的：

| 参数 | 目的 |
|---|---|
| `--trace=cuda,nvtx,osrt` | 关联 CUDA launch、NVTX 与部分 OS runtime 等待 |
| `--sample=none --cpuctxsw=none` | 首轮不依赖 CPU sampling/context-switch 权限 |
| `--trace-fork-before-exec=true` | 跟踪进程树；同时使用 spawn 降低 fork 初始化问题 |
| `--cuda-graph-trace=node` | 看单个 kernel；当前 eager 不依赖它，后续 graph 对照可复用 |
| `--capture-range=cudaProfilerApi` | 等 worker 的 cudaProfilerStart 才记录正式窗口 |
| `--capture-range-end=stop` | 第一次停止后不再收集后续请求 |
| `--kill=none` | 不在采集结束时强制杀模型服务 |
| `-o` | 独立报告前缀，避免覆盖历史报告 |

不同版本对进程退出/报告刷新的时机有差异；`stop` 表示停止 collection，不保证长驻服务自行退出。回放结束后，回到终端 A 按 Ctrl+C，等 nsys 正常写完报告，不用 `kill -9`。同一个服务下一次采集要重新启动；多窗口需另选 `repeat` 语义，首轮不使用。

此模式依据 [vLLM 0.14.1 profiling 文档](https://docs.vllm.ai/en/v0.14.1/contributing/profiling/)；参数语义以 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)及已保存的本地 help 为准。

### 8.2 终端 B：等待服务、预热、触发并回放 1 个请求

激活相同 venv，然后 `source` 终端 A 输出目录内的文件。**填入终端 A 的实际 OUT，不能重新执行时间戳命令生成另一个 OUT。**

```bash
source /workspace/venvs/vllm-0112/bin/activate
# 将这里替换为终端 A 的真实文件路径：
source /workspace/AI_system/out/quantization_profiling/nsight_实际时间/session.env
export PROFILE=fp8-local
export CASE=w1024_o128
export PORT=7081
curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/health"
python "$OUT/replay_http.py" \
  --requests-file "$REQUESTS" --case "$CASE" --profile "$PROFILE" \
  --count 1 --base-url "http://127.0.0.1:$PORT" --control \
  --output "$OUT/nsys/${PROFILE}_${CASE}_client.jsonl"
```

如果 health 还没成功，继续等终端 A 的启动日志；不要在加载中直接启动正式回放。运行中应看到 5 次 warmup 成功，然后 `/start_profile` 成功、1 次 measure 成功、`/stop_profile` 成功。失败则先修复请求，不生成性能归因。控制接口 HTTP 200 只证明调用被接收；worker wrapper 可能记录并吞掉底层 profiler start/stop 异常，因此还必须检查 worker 日志中的 `Failed to start/stop profiler`，并确认报告内确实存在正式窗口的 CUDA 轨迹。

### 8.3 四模型依次采集

每次终端 A 退出、确认该服务释放显存后，再启动下一组；终端 B 必须设置一致 PROFILE/CASE。

| 顺序 | PROFILE | CASE | 目的 |
|---:|---|---|---|
| 1 | fp8-local | w1024_o128 | 找本地 FP8 decode 主成本 |
| 2 | fp8-official | w1024_o128 | 判断是否相同热点/调用次数 |
| 3 | bf16 | w1024_o128 | 找 FP8 新增的 quant/scale 与时间差 |
| 4 | awq | w1024_o128 | 对照 Marlin、小 M 效率 |
| 5～6 | 两份 FP8 | w4096_o256 | 看 chunked prefill 与 decode 的差异 |
| 7～8 | BF16 / AWQ | w4096_o256 | 给长输入建立相同条件的参照 |
| 按需 | 四组 | w8192_o128 / w1024_o512 | attention/KV 与长 decode 尾部 |

命令只需替换：

```bash
export PROFILE=fp8-official
export CASE=w4096_o256
```

再重复 8.1 / 8.2 的采集命令；不要复用旧报告前缀来覆盖证据。若输出文件已存在，换独立试验目录或增加 `_r2` 后缀，同时修改客户端文件名。

### 8.4 没抓到 worker 的降级检查

```bash
ps -eo pid,ppid,comm,args --forest | grep -E 'vllm|EngineCore|Worker|nsys'
grep -iE 'profil|error|warning|quant|dtype|triton|marlin|cutlass' \
  "$OUT/nsys/${PROFILE}_${CASE}_server.log"
```

目的：确认 nsys 包住的进程树里真有 GPU worker。HTTP client 的网络请求正常，不代表 worker 已被注入或触发。

如果 CUDA profiler 控制不可用，可暂时从启动开始采集，然后只分析预热后窗口：去掉两个 `--capture-range*` 参数，保留其余选项，回放时去掉 `--control`，最后正常停止服务。该降级会捕获模型加载/JIT/预热，报告较大；必须在 GUI 选出正式请求，不能用整份统计直接比较推理。

## 9. Step 6：从 nsys 报告得到可解释结论

### 9.1 CLI：先找热点和调用次数

```bash
export PROFILE=fp8-local CASE=w1024_o128
export REP="$OUT/nsys/${PROFILE}_${CASE}_eager.nsys-rep"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$REP" \
  > "$OUT/evidence/${PROFILE}_${CASE}_summary.txt"
nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum "$REP" \
  > "$OUT/evidence/${PROFILE}_${CASE}_memory.txt"
nsys stats --report cuda_gpu_kern_sum --format csv "$REP" \
  > "$OUT/evidence/${PROFILE}_${CASE}_kernels.csv"
```

report ID 若不存在，按第 4 节的 `nsys-reports.txt` 选择当前版本。CLI CSV 可能带 report 标题行，写解析器前先看文件前几行。

如何看：

| 观察 | 要追问什么 |
|---|---|
| `*_block_scaled_mm*` 或 FP8 GEMM 占比大 | M=1 kernel 本身是不是比 BF16 对应 GEMM 更慢？ |
| group quant / scale / cast 很多 | 每层新增几次、每 token 累计多少？ |
| Marlin 高效，FP8 小 kernel 多 | 是否为不同融合和 tile 设计造成？ |
| CUDA API 时间高 | 是 CPU 发射开销，还是同步在等待已有 GPU 工作？ |
| memcpy/memset/malloc 频繁 | 拷贝方向、尺寸、是否在每步重复？ |
| 大量 GPU 空洞 | scheduler/IPC/Python/同步是否阻塞了下一个 launch？ |

`Time (%)` 通常以该报告中 kernel 累计时间为分母，不是整个请求 E2E；同步 API 时间可与 kernel 时间重叠。不要把 `cudaStreamSynchronize` 的墙钟与 GPU 时间相加。

### 9.2 GUI：先定位 GPU worker，再分阶段

将 `.nsys-rep` 拷到本地后用 Nsight Systems GUI 的 File → Open 打开。

1. 展开进程树，找到实际执行 CUDA 的 EngineCore / worker，记录 PID 与 GPU UUID。
2. 展开 `CUDA HW` 的 Kernels、Memory 和对应线程的 CUDA API / NVTX。API frontend 主要做 HTTP，不一定拥有 CUDA context。
3. 找到正式窗口第一段较大的计算。1024 输入、2048 budget、并发 1 时通常是一个 prefill step；4096/8192 输入可能有多个 chunk，须由 scheduler/shape 证实。
4. 后面重复的小周期通常对应 decode step。vLLM 0.14.1 在 CUDA profiler 中已有 worker 标记，可能看到类似 `execute_context_1(1024)_generation_0(0)`（context）和 `execute_context_0(0)_generation_1(1)`（generation）；名称和计数以实际 trace 为准，chunked/mixed step 不能机械套这个示例。挑中间稳定的 16～32 步，同时避开首次编译/首次特定 shape 和结束清理。
5. 框选这段时间，查看该 selection 的 kernel 统计、调用次数、总时间。GUI 不同版本的上下文菜单名称不同，核心是只对选中窗口统计。
6. 点一个 FP8 GEMM，记录 kernel 完整名、duration、grid、block、stream、correlation 与父 NVTX。
7. 通过 correlation 回看 CPU launch；查看 launch 到执行之间的等待、前后 GPU 空洞，而非只看 kernel 长度。
8. 对官方 FP8、BF16、AWQ 重复相同阶段/相同 token 位置，截图并填下表。

| 模型 | 阶段/step 范围 | 窗口 us | GPU busy 区间并集 us | kernel 累计 us | launch 数 | quant/scale us | GEMM us | attention us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| local FP8 | 待填 | | | | | | | |
| official FP8 | 待填 | | | | | | | |
| BF16 | 待填 | | | | | | | |
| AWQ | 待填 | | | | | | | |

并发 stream 可能重叠：GPU busy 要对 `[start,end]` 求并集；kernel 累计时间可大于 busy。`窗口时间 - busy` 只是该 GPU 上未覆盖 kernel 的时间，还应考虑 memcpy 等 engine 活动，不能直接全部算 CPU 调度成本。

### 9.3 prefill / decode 的严格区分

仅靠“前面大、后面小”可用于初筛；严格归因要有 scheduler step、模型 forward token 行数、torch profiler shape 或 worker NVTX 佐证。请求包含首 token 采样，prefill 输出1的诊断也不等于纯 prefill CUDA 时间。

不要将所有 CUDA kernel 的第 N 次调用直接称为第 N 个 token。一次 forward 有多层和多种 kernel，chunked prefill 又可能占多个 step。

### 9.4 SQLite：需要批量统计时再用

```bash
nsys export --type sqlite \
  --output "$OUT/evidence/${PROFILE}_${CASE}.sqlite" "$REP"
```

先检查 schema，因为 Nsight 版本会改变列：

```bash
python - <<'PY'
import os, sqlite3
from pathlib import Path
p = Path(os.environ['OUT'])/'evidence'/f"{os.environ['PROFILE']}_{os.environ['CASE']}.sqlite"
c = sqlite3.connect(p)
for table, sql in c.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"):
    if any(s in table for s in ('KERNEL','MEMCPY','NVTX','StringIds')):
        print(table, sql)
PY
```

常见 kernel 表为 `CUPTI_ACTIVITY_KIND_KERNEL`，时间为 ns。确认存在 `start,end` 后，可按**已经从 GUI 确认的**窗口取完全落在其中的 kernel：

```sql
SELECT COUNT(*) AS launches,
       SUM(end - start) / 1000.0 AS summed_kernel_us
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start >= :window_start_ns AND end <= :window_end_ns;
```

目的：同一窗口按名称/次数批量对比；name 列有时是 `StringIds` 的 ID，需要按实际 schema join。不要直接把全报告 SQL 总和叫 decode 时间，也不要忽略边界跨越的 kernel 或多 stream 重叠。

### 9.5 可选：设备级 GPU metrics

仅当首轮时间线仍不能判断且权限/硬件支持时，额外一轮在 8.1 添加：

```text
--gpu-metrics-devices=all
```

用 `nsys profile --help` 确认采样频率选项后再调整频率。该指标可能是设备级，混入其它进程；不能替代 NCU 的单 kernel 分析。采 GPU metrics 的这轮结果单独保存，检查开销，不与无 profiler 吞吐混算。

## 10. Step 7：准备 NCU 专用离线入口，排除 HTTP 与网络

使用相同模型、冻结 token IDs、BF16 激活、KV dtype、调度 budget、固定输出与 warmup；让生成在同一进程树内自动开始/结束，便于 replay。离线输出不是原 HTTP benchmark，不能拿其 token/s 覆盖用户报告。

### 10.1 生成脚本

```bash
cat > "$OUT/profile_offline.py" <<'PY'
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from model_profiles import resolve_model_alias, validate_model_profile, resolve_runtime_options


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=('bf16', 'awq', 'fp8-official', 'fp8-local'), required=True)
    p.add_argument('--requests-file', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--output-tokens', type=int)
    p.add_argument('--batch-tokens', type=int, default=2048)
    p.add_argument('--graph', action='store_true')
    p.add_argument('--profiler', choices=('none', 'cuda', 'torch'), default='cuda')
    p.add_argument('--delay-iters', type=int, default=0)
    p.add_argument('--max-iters', type=int, default=0)
    p.add_argument('--torch-dir', type=Path)
    p.add_argument('--unique-output', action='store_true', help='unique diagnostic JSON per replay process')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.count, a.batch_size, a.warmup, a.batch_tokens) < 1:
        p.error('count, batch-size, warmup, batch-tokens must be positive')
    if a.output_tokens is not None and a.output_tokens < 1:
        p.error('output-tokens must be positive')
    if min(a.delay_iters, a.max_iters) < 0:
        p.error('iteration gates must be nonnegative')
    if a.profiler == 'torch' and a.torch_dir is None:
        p.error('torch profiler requires --torch-dir')
    if a.unique_output:
        a.output = a.output.with_name(a.output.stem + '_' + str(os.getpid()) + a.output.suffix)
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = [json.loads(s) for s in a.requests_file.read_text().splitlines() if s.strip()]
    def select(stage):
        return sorted((r for r in rows if r['case'] == a.case and r['phase'] == stage),
                      key=lambda r: r['id'])
    warm, measured = select('warmup'), select('measure')
    if len(warm) < 5 or len(measured) < a.count * a.batch_size:
        raise ValueError('insufficient frozen requests')
    sample = measured[0]
    out_tokens = a.output_tokens or sample['output_tokens']
    # Changing only generated tokens must not silently change the engine length limit.
    max_len = sample['input_tokens'] + sample['output_tokens']
    if sample['input_tokens'] + out_tokens > max_len:
        raise ValueError('diagnostic output exceeds the original engine length limit')
    cfg_file = Path(os.environ['DEMO']) / 'configs' / {
        'bf16': 'bf16.yaml', 'awq': 'awq.yaml',
        'fp8-official': 'fp8-official.yaml', 'fp8-local': 'fp8.yaml',
    }[a.profile]
    cfg = json.loads(cfg_file.read_text())
    model = resolve_model_alias(a.profile)
    model_cfg = json.loads((Path(model)/'config.json').read_text())
    validate_model_profile(model_cfg, a.profile)
    options = resolve_runtime_options(model_cfg, cfg['dtype'], cfg.get('quantization') or 'auto')
    kwargs = dict(model=model, tokenizer=cfg.get('tokenizer') or model,
                  dtype=options['dtype'], tensor_parallel_size=1,
                  max_model_len=max_len, gpu_memory_utilization=cfg['gpu_memory_utilization'],
                  max_num_seqs=a.batch_size, max_num_batched_tokens=a.batch_tokens,
                  enable_chunked_prefill=True, enable_prefix_caching=False,
                  kv_cache_dtype='auto', enforce_eager=not a.graph,
                  seed=42, generation_config='vllm')
    if options['quantization'] != 'auto':
        kwargs['quantization'] = options['quantization']
    pcfg = None
    if a.profiler != 'none':
        pcfg = dict(profiler=a.profiler, delay_iterations=a.delay_iters,
                    max_iterations=a.max_iters)
        if a.profiler == 'torch':
            pcfg.update(torch_profiler_dir=str(a.torch_dir.resolve()),
                        torch_profiler_record_shapes=True,
                        torch_profiler_with_stack=True,
                        torch_profiler_with_memory=False,
                        ignore_frontend=True)
        kwargs['profiler_config'] = pcfg
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from vllm import LLM, SamplingParams
    print(json.dumps({'engine': kwargs, 'diagnostic': vars(a)}, default=str), flush=True)
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0, top_p=1, top_k=-1,
                            repetition_penalty=1, seed=42, ignore_eos=True,
                            min_tokens=out_tokens, max_tokens=out_tokens)
    def generate(items):
        for r in items:
            if len(r['prompt_token_ids']) != sample['input_tokens']:
                raise ValueError('frozen input length mismatch')
        outputs = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in items],
                               sampling_params=params, use_tqdm=False)
        if len(outputs) != len(items):
            raise ValueError('output request count mismatch')
        for result in outputs:
            if len(result.outputs[0].token_ids) != out_tokens:
                raise ValueError('fixed output length mismatch')
        return outputs
    for i in range(a.warmup):
        generate([warm[(i+j) % len(warm)] for j in range(a.batch_size)])
    records = []
    if a.profiler != 'none':
        llm.start_profile()
    try:
        for i in range(a.count):
            items = measured[i*a.batch_size:(i+1)*a.batch_size]
            start = time.perf_counter()
            outputs = generate(items)
            records.append({'ids': [r['id'] for r in items],
                            'wall_seconds': time.perf_counter()-start,
                            'output_tokens': [len(r.outputs[0].token_ids) for r in outputs]})
    finally:
        if a.profiler != 'none':
            llm.stop_profile()
    report = {'profile': a.profile, 'case': a.case, 'engine': kwargs,
              'args': vars(a), 'records': records,
              'versions': {k: importlib.metadata.version(k) for k in ('vllm','torch','triton')},
              'requests_file_sha256': hashlib.sha256(a.requests_file.read_bytes()).hexdigest(),
              'timing_note': 'offline diagnostic; profiler/replay overhead possible; not formal HTTP benchmark'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
PY
python -m py_compile "$OUT/profile_offline.py" "$OUT/replay_http.py"
```

目的：主程序以 `if __name__ == '__main__'` 保护，兼容 spawn；不在 client 上错误开启 CUDA context；通过 `LLM.start_profile()` / `stop_profile()` 控制真正 worker。相关接口见 [v0.14.1 LLM 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/entrypoints/llm.py)。

脚本的模型路径由 `MODEL_ROOT` 和项目 alias 生成；如果原实验用自定义模型路径或 tokenizer，先按原 run.json 修改这个诊断脚本。首轮 batch=1 与原协议对应；batch>1 是额外实验，预热请求可能在 batch 内循环复用。

### 10.2 无 Nsight 的离线对照

先停止 HTTP 模型服务，再执行：

```bash
python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --profiler none --count 3 \
  --output "$OUT/evidence/local_offline_no_profiler.json"
```

对其余三个模型重复。目的：确认离线也复现 FP8 慢。如果差距消失，回到 HTTP/scheduler/IPC/流式处理调查；若仍存在，GPU 计算路径更值得优先分析。不要把 3 个请求的结果做 p95 结论。

### 10.3 用 nsys 给离线入口做交叉验证

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUT/nsys/local_offline_w1024_o128" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --output "$OUT/evidence/local_offline_nsys.json"
```

确认离线和 HTTP 的主要 kernel 家族、shape、调用次数大体一致，再用它代表 HTTP 服务做 NCU kernel 调查。离线包含相同 engine 核心，但不包含完整 HTTP frontend 代价。

## 11. Step 8：Nsight Compute 从一个热点开始

**nsys、ncu、torch GPU profiler 分别运行，不套娃、不并发。** NCU 为计数器 replay，可能把一个正常几秒的请求放大很多倍；不对完整 80 请求做 `--set full`。

### 11.1 先选热点，而不是盲猜 kernel 名

从第 9 节选择三类候选：

1. FP8 block GEMM：至少一个 decode M=1 shape；再选一个 prefill shape。
2. 动态 activation quant / scale：确认每 token 累计成本。
3. 一个 attention 或其它高占比 kernel：解释不随权重量化下降的公共成本。

BF16、AWQ 各选相应逻辑层的 GEMM。真实函数名包含模板参数、tile 和特化后缀；先用宽 regex 找到，再缩到特定 kernel/调用。下面 regex 是搜索示例，必须按实际 nsys 名称调整。

### 11.2 本地 FP8：先抓一个 prefill GEMM

```bash
export KERNEL_RE='.*(w8a8.*block.*|block.*scaled.*mm).*'
ncu --target-processes all \
  --profile-from-start off \
  --set basic --replay-mode kernel \
  --kernel-name-base demangled --kernel-name "regex:$KERNEL_RE" \
  --launch-count 1 \
  --export "$OUT/ncu/local_prefill_gemm_basic" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --output-tokens 1 \
  --output "$OUT/evidence/local_prefill_ncu_basic.json"
```

目的：输出只生成 1 token，消除后续长 decode，只抓预热后的第一个匹配 GEMM。仍包含 prefill、logits、采样等其它计算；NCU 的 kernel filter 才把观测收窄到所选 GEMM。

`--profile-from-start off` 避开模型加载/预热，等待 worker 的 CUDA start。`--target-processes all` 包含 GPU worker。`--launch-count 1` 是匹配的 kernel launch 个数，不是请求数或 token 数。

### 11.3 本地 FP8：只取稳定 decode 窗口

```bash
ncu --target-processes all \
  --profile-from-start off \
  --set basic --replay-mode kernel \
  --kernel-name-base demangled --kernel-name "regex:$KERNEL_RE" \
  --launch-count 2 \
  --export "$OUT/ncu/local_decode_gemm_basic" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_ncu_basic.json"
```

目的：用 vLLM worker 的 `delay_iterations` 跳过最初若干 engine step，再采集 2 个 step 中的少量热点。**iteration 是 engine step，不是 token 下标；** chunked prefill、batch 和 wrapper 调用位置会影响边界。在已核对的 vLLM 0.14.1 wrapper/worker 实现中，`delay=8,max=2` 会在第 8 次 step 的 forward 前开始，采第 8、9 次 forward，在第 10 次 step 的 forward 前停止；严格说跳过前 7 次，而非 8 次。预热时 profiler 未 active，不计入这个延迟。先将相同参数放入 10.3 的 nsys 命令，确认窗口确实位于 decode，再用 NCU。边界依据：[v0.14.1 GPU worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/v1/worker/gpu_worker.py)与前述 profiler wrapper。

`--max-iters 2` 与 `--launch-count 2` 含义不同：前者控制 worker step 范围，后者限制匹配 kernel 数。只看到 2 个 launch 不代表抓了 2 个完整 token。

### 11.4 官方 FP8、BF16、AWQ 的对应采集

官方 FP8 使用同类 FP8 regex，改为：

```text
--profile fp8-official
--export .../official_decode_gemm_basic
--output .../official_decode_ncu_basic.json
```

BF16 先从 nsys 找真实 GEMM 名，再设置例如：

```bash
export KERNEL_RE='.*(gemm|gemv|xmma).*'
```

AWQ 同理，例如：

```bash
export KERNEL_RE='.*[Mm]arlin.*'
```

然后复用 11.3，分别改成 `--profile bf16` / `--profile awq`、不同报告路径。比较的是**同一模型逻辑投影、同 M/N/K 的实现**，不能比较 FP8 qkv 与 BF16 down_proj。

### 11.5 只抓 quant kernel

```bash
export QUANT_RE='.*(per_token_group.*quant|group.*fp8.*quant|scaled_fp8_quant).*'
ncu --target-processes all --profile-from-start off \
  --set basic --kernel-name-base demangled \
  --kernel-name "regex:$QUANT_RE" --launch-count 1 \
  --export "$OUT/ncu/local_decode_quant" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_quant.json"
```

目的：判断动态量化是少量额外成本，还是高频显著热点。若实际 kernel 已融合进其它 op，不应因为 regex 0 命中就认定没有量化。

### 11.6 更准确的 launch 选择

- `--launch-skip N` 跳过前 N 个**匹配过滤器**的 kernel；不是跳过 N 个请求。
- `--launch-skip-before-match N` 按所有 launch 计数，版本支持时才用。
- `--kernel-id` 可约束 context/stream/name/invocation；格式以本地 `ncu --help` 为准。
- decode 中相同名字可能有多个层、grid、shape；结合 nsys correlation、NCU launch stats、op shape 选定。
- 延迟 gate 比估算“前面有多少个层×kernel”更稳，但仍需 nsys 确认阶段。

如果报告显示 `No kernels were profiled`，先放宽 regex、确认 CUDA gate 与 worker 注入，不能直接得出 GPU 没工作。CLI 语义参考 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

## 12. Step 9：加硬件指标，解释热点为什么慢

### 12.1 按问题选择 section

先核对 `ncu --list-sections`。确认以下 ID 存在后，对已经选准的一个 decode GEMM 加深分析：

```bash
export KERNEL_RE='替换为已经命中的目标GEMM正则'
ncu --target-processes all --profile-from-start off \
  --replay-mode kernel --kernel-name-base demangled \
  --kernel-name "regex:$KERNEL_RE" --launch-count 1 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --export "$OUT/ncu/local_decode_gemm_deep" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_ncu_deep.json"
```

| Section | 看什么 | 如何用于本问题 |
|---|---|---|
| SpeedOfLight | SM 与 memory 吞吐占峰值比例 | 初筛资源利用与限制，不能单独下结论 |
| LaunchStats | grid/block、registers、shared memory、waves/SM | M=1 是否规模不足、tile 浪费、资源占用过高 |
| Occupancy | 理论/实际 occupancy、限制因素 | 是寄存器/shared memory 限制，还是任务太少 |
| MemoryWorkloadAnalysis | DRAM/L2 流量、带宽、命中率 | 权重流量是否减少，有没有额外中间读写 |
| ComputeWorkloadAnalysis | 指令管线、Tensor 等利用 | 是否使用/受限于计算管线 |

进一步有明确问题才加 `SchedulerStats`、`WarpStateStats`、`SourceCounters`、`InstructionStats` 或该版本的 roofline section。一次加太多 section 会显著增加 replay 次数。

### 12.2 只采少量 metric 的轻量命令

先查询目标卡支持的完整后缀：

```bash
ncu --query-metrics --query-metrics-mode all \
  > "$OUT/evidence/metric-full-names.txt"
grep -E 'gpu__time_duration|dram__bytes|dram__throughput|sm__throughput|pipe_tensor|warps_active|lts__t_bytes' \
  "$OUT/evidence/metric-full-names.txt"
```

常见候选：

```text
gpu__time_duration.sum
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
```

Tensor Core 指标的名称/后缀跨架构变化，按 `pipe_tensor` 的实际查询结果选择，不能照搬 Hopper 的某个 metric 到 Ada。

确认候选都存在后，例如只采时间与 DRAM 流量：

```bash
ncu --target-processes all --profile-from-start off \
  --kernel-name-base demangled --kernel-name "regex:$KERNEL_RE" \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  --export "$OUT/ncu/local_decode_gemm_bytes" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_ncu_bytes.json"
```

目的：控制采集量，回答“FP8 到底省了多少 DRAM 流量，是否被额外开销抵消”。读字节减少不保证时间等比例减少。

### 12.3 导出报告和 SASS

```bash
ncu --import "$OUT/ncu/local_decode_gemm_deep.ncu-rep" --page details \
  > "$OUT/evidence/local_decode_details.txt"
ncu --import "$OUT/ncu/local_decode_gemm_deep.ncu-rep" --page raw --csv \
  > "$OUT/evidence/local_decode_metrics.csv"
ncu --import "$OUT/ncu/local_decode_gemm_deep.ncu-rep" --page source --print-source sass \
  > "$OUT/evidence/local_decode_sass.txt"
```

目的：形成可审阅证据。没有源码/line info 时，SASS 或源码关联可能不完整；Triton JIT 的 Python 源行也不保证自动出现。

在 Nsight Compute GUI：File → Open 打开 `.ncu-rep`；对基准结果设 baseline，再比较同阶段/shape 的结果。不同精度 kernel 名称可以不同，仍按模型逻辑工作量配对。

### 12.4 如何确认是真正 FP8 Tensor Core 计算

需要三类证据：

1. checkpoint 与实际 op 的输入/权重/scale dtype、scheme 和 shape。
2. 目标 kernel 的 SASS / 指令信息与对应架构的矩阵指令类型。
3. Tensor 管线活动，以及对应 GEMM 的时间与流量。

不要只搜函数名中的 `fp8`。SM89 上不能要求一定看到 Hopper 的 `WGMMA`；矩阵指令的名字与类型编码随架构而变。看到 `HMMA` 等矩阵指令后仍要核对操作数精度；看到 conversion 指令也不等于整层退回 BF16。Triton 的 `tl.dot` 能走 Tensor Core，是否达到高效率是另一个问题。

### 12.5 六种结果模式与下一步

| 证据模式 | 可能解释 | 下一步 |
|---|---|---|
| FP8 DRAM bytes 更少，但 GEMM duration 更长；waves 低 | 小 M tile/并行度或分块处理不利 | 对照 prefill、大 batch 与对应 shape |
| GEMM 比 BF16 快，但 quant/scale 总成本很大 | 动态激活量化抵消收益 | 按 token 累计成本，检查融合机会 |
| 单 kernel 很快，GPU 周期大量空洞 | host launch/scheduler/同步主导 | eager vs graph；必要时 CPU sampling |
| FP8 长 prefill 快、decode 慢 | 大/小 M 后端效率差异 | 分开报告 prefill 与 decode |
| 两份 FP8 热点与次数相同，但时间差随轮次变 | 运行波动/时钟/其它进程可能较大 | 无 profiler 多轮、交错顺序复测 |
| 两份 FP8 的 op/shape/kernel 不一致 | loader、scheme、忽略层或布局不同 | 对照 config、source、实际层 dispatch |

“低 DRAM 带宽+低 SM”不自动说明 CPU bound：也可能是小任务、访存延迟、依赖或低并行度；必须结合 nsys gaps 和单 kernel 指标。

### 12.6 Roofline 与算术强度：只用于对应工作量

稠密 GEMM 的逻辑运算量约为 `2*M*N*K`。算术强度可估为 `逻辑 FLOPs / 实测 DRAM bytes`，同时注意 block scaling、padding、其它计算与多 kernel。FP8 与 BF16 要用各自对应精度的硬件峰值，不拿不同精度的 `%peak` 数字直接判谁更快。

decode M=1 常常重读大量权重，低精度有带宽收益的可能；但高带宽并不充分证明这是唯一瓶颈。NCU 的实际访存量还受 replay/cache 设置影响。方法依据：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 13. Step 10：NCU replay、CUDA Graph 与 NVTX 的进阶用法

### 13.1 replay 模式的选择

| 模式 | 作用 | 本项目建议 |
|---|---|---|
| `kernel` | 对匹配的单 kernel 多次 replay | 首轮单 GPU eager 热点首选 |
| `application` | 整个应用重新启动，每次收不同指标 | 用能自动退出的 offline，不能指望外部 HTTP 自动重放 |
| `range` | 捕获/重放一个 CUDA API/kernel 范围 | 支持的 API 有限制，保持范围很短 |
| `app-range` | 重启应用，按范围整体采计数器 | 版本支持且需要保留范围并发语义时再试 |

application 示例，复用已选中的 kernel：

```bash
ncu --target-processes all --profile-from-start off \
  --replay-mode application \
  --kernel-name-base demangled --kernel-name "regex:$KERNEL_RE" \
  --launch-count 1 --set basic \
  --export "$OUT/ncu/local_decode_app_replay" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --unique-output --output "$OUT/evidence/local_decode_app_replay.json"
```

此例用 helper 的 `--unique-output` 为每次 replay 的诊断 JSON 附加 PID，避免第二次启动撞同名文件。`.ncu-rep` 仍由 NCU 统一输出。application replay 重载 8B 模型成本高，优先减少 metric 数避免需要它。

`range` / `app-range` 示例，只在已经核实 worker gate 范围为 1～2 decode step 且本地 help 支持时使用：

```bash
ncu --target-processes all --profile-from-start off \
  --replay-mode range --set basic \
  --export "$OUT/ncu/local_decode_range" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_range.json"
```

这是范围指标，不是每个 kernel 的单独报告。遇到 unsupported API / 跨进程限制，就回到单 kernel 的最小采集，不用不断扩大范围。多 GPU/NCCL 还有必须并发的 kernel 与 replay 死锁风险；本项目 TP=1 首轮不引入这些变量。

### 13.2 CUDA Graph 先用 Systems 验证，再考虑 Compute

首轮全部 eager 是为了复现原报告；Graph 消融时把 HTTP 启动器 `EAGER=0`，或 offline 加 `--graph`。

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUT/nsys/local_decode_graph_nodes" \
  python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 --graph \
  --output "$OUT/evidence/local_graph_nsys.json"
```

目的：确认 decode 真出现 graph launch，CPU launch 数和 GPU gaps 是否下降。单纯去掉 eager flag 不保证每个 shape 都被 graph 覆盖；启动日志和时间线需要同时确认。Graph 初始化、编译、capture 都在预热/正式窗口外。

需要 NCU 时，先检查当前版本的 `--graph-profiling node|graph`、replay 兼容性。`node` 拆节点，可能改变原并发/cache 行为；`graph` 把图作为整体，部分 source 指标不可用。不要把 graph 模式计数器和 eager 单 kernel 指标无条件逐项相减。具体兼容规则参见上节 NVIDIA Profiling Guide 的 Graph Profiling。

### 13.3 NVTX 的正确位置与过滤语法

NVTX 必须在实际发射 CUDA 的 worker 线程中标记。下面是需要插入实际 forward 的示意，**不是独立运行脚本**：

```python
with torch.cuda.nvtx.range('qkv_proj'):
    qkv, _ = self.qkv_proj(hidden_states)
```

更有用的标记包括 step 编号、num_tokens、layer、投影名；只在少量诊断请求启用，记录代码 patch，不把逐层 Python instrumentation 的耗时当正式性能。

确认 GUI 中 worker 真的有这个 push/pop range 后，可在 NCU 命令上加：

```text
--nvtx --nvtx-include 'qkv_proj/'
```

自定义 domain 示例：

```text
--nvtx --nvtx-include 'model@qkv_proj/'
```

push/pop 单范围末尾 `/` 有语法意义；`nvtxRangeStart/End` 的过滤不加这个末尾 `/`。不要把带斜杠的显示名称误当实际嵌套。HTTP parent 的 NVTX range 不会跨进程覆盖 worker。

nsys 还支持 NVTX 触发窗口，例如已有 worker 标记 `profiled_decode` 时。若使用 `torch.cuda.nvtx.range()` 这类普通字符串，先允许非 registered string 匹配（默认可能只匹配 registered strings）：

```bash
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
```

然后在 nsys 命令中使用：

```text
--capture-range=nvtx --nvtx-capture=profiled_decode
```

这是替换 cudaProfilerApi 触发方式，不是随便叠两个触发器；NVTX 专项实验结束后可 `unset NSYS_NVTX_PROFILER_REGISTER_ONLY`。Graph replay 中 NVTX 的投影与 capture 时机依赖工具版本，首轮优先使用 worker CUDA profiler gate。

### 13.4 cache 与 clock 控制要记录

查看本地 NCU help 的 `--cache-control`、`--clock-control`。默认值跨版本可能变化，四模型比较时记录并保持一致。

- 默认 cache flush 往往更可重复，但与连续 decode 的自然 cache 状态有差异。
- `--cache-control none` 更接近保留 cache，但多 pass 的数值可能受前一次 replay 影响。
- `--clock-control none` 保留自然时钟，并不等于时钟固定。
- 不把 profiler 控制时钟下的 kernel duration 当原 benchmark 的绝对时延；与 nsys/无 profiler 交叉检查。

不建议本轮直接用锁频/功耗限制调参掩盖根因，先解释现有路径。

## 14. Step 11：把 kernel 映射回 Qwen3 线性层

### 14.1 先计算真实 M/N/K 候选

```bash
python - <<'PY' > "$OUT/evidence/logical-gemm-shapes.txt"
import json, os
from pathlib import Path
c = json.loads((Path(os.environ['MODEL_ROOT'])/'Qwen3-8B/config.json').read_text())
h, inter = c['hidden_size'], c['intermediate_size']
heads = c['num_attention_heads']
kv_heads = c.get('num_key_value_heads', heads)
d = c.get('head_dim', h // heads)
print('TP=1; W=[N,K]; input=[M,K]; num_layers=', c['num_hidden_layers'])
for name, n, k in (
    ('qkv_proj', (heads+2*kv_heads)*d, h),
    ('o_proj', h, heads*d),
    ('gate_up_proj', 2*inter, h),
    ('down_proj', h, inter),
):
    print(name, 'N=', n, 'K=', k, 'decode M typically=1; prefill M determined by scheduler')
PY
```

目的：识别投影的逻辑尺寸。vLLM 常合并 Q/K/V，合并 gate/up；磁盘上七个权重不代表七个独立 forward GEMM。实际 kernel 还可能采用转置、padding、split-K、融合或分 shard，最终以 op 参数/trace 为准。

### 14.2 单独运行 torch profiler，补 op/shape/stack

先停止其它 profiler 和模型服务，再执行：

```bash
python "$OUT/profile_offline.py" \
  --profile fp8-local --requests-file "$REQUESTS" --case w1024_o128 \
  --profiler torch --torch-dir "$OUT/torch/local_decode" \
  --delay-iters 8 --max-iters 2 \
  --output "$OUT/evidence/local_decode_torch.json"
```

对官方 FP8、BF16、AWQ 分别执行，换独立 torch-dir 和 output。`record_shapes=True` 与 stack 有额外开销，只用于结构映射。

```bash
find "$OUT/torch" -type f \
  \( -name '*.pt.trace.json*' -o -name 'profiler_out_*.txt' \) -print
```

用本地 trace viewer 打开 worker trace，沿 CPU op → CUDA correlation 查：

- FP8 block quant / block scaled GEMM 的 op，输入 shape 与 scale shape。
- prefill 的 M 与 decode 的 M 是否分别为大 token 行数和 1。
- BF16 GEMM、AWQ-Marlin 是否对应同一投影。
- lm_head、attention、norm、rotary、sampling 等是否仍以公共路径消耗显著时间。

torch.compile/custom op 可能使内部 shape 或 Python stack 不完整；这时结合第 6 节源码与 NCU launch 信息，不把缺失 shape 强行补成猜测。当前 eager 首轮也更容易建立映射。

### 14.3 local 与 official 的具体核对表

| 对比项 | local | official | 结论如何用 |
|---|---|---|---|
| source base revision / 模型结构 | 填实际值 | 填实际值 | 排除不同模型 |
| loader / quant scheme | | | loader 不同可能汇合到同一 LinearOp |
| 权重分块与 activation group | | | 检查是否都是 128×128 / group128 |
| ignored modules | | | 是否有不同的 BF16 层 |
| qkv/gate_up 合并与 scale 布局 | | | 是否引入额外转换 |
| prefill 实际 kernel、M/N/K | | | 验证后端是否一致 |
| decode 实际 kernel、grid/block | | | 验证小 M 特化是否一致 |
| 每 step quant/GEMM/其它 launch 数 | | | 找额外 op 或融合差异 |
| 相同窗口 GPU busy / gap | | | 区分 GPU 与 host 差异 |
| NCU DRAM、Tensor、资源、耗时 | | | 对应硬件原因 |

吞吐接近但 E2E p95 某一组差 10% 并不矛盾：吞吐与尾部是不同统计量，也可能有单轮异常。先读逐请求分布与遥测，再决定是否需要 kernel 深挖。

## 15. Step 12：受控消融，验证原因而不是只画图

每次只改一类因素，local 与 official 优先做，BF16/AWQ 保留同条件参照。任何收益最终都回到**无 profiler**环境复测。

### 15.1 eager → CUDA Graph：验证 host launch 开销

从配置复制到诊断目录，不改原 config：

```bash
mkdir -p "$OUT/configs"
python - <<'PY'
import json, os
from pathlib import Path
root, out = Path(os.environ['DEMO']), Path(os.environ['OUT'])/'configs'
for profile, file in (('fp8-local','fp8.yaml'), ('fp8-official','fp8-official.yaml'),
                      ('bf16','bf16.yaml'), ('awq','awq.yaml')):
    cfg = json.loads((root/'configs'/file).read_text())
    cfg['enforce_eager'] = False
    (out/f'{profile}-graph.json').write_text(json.dumps(cfg, indent=2)+'\n')
PY
python "$DEMO/scripts/benchmark.py" \
  --config "$OUT/configs/fp8-local-graph.json" --case w1024_o128 \
  --requests 80 --rounds 3
```

对其余模型换 config 重复。新实验 ID 会记录新的有效协议。原冻结快照只有 1 轮并不妨碍选择多轮重复相同输入，脚本将有效 rounds 单独保存；正式请求数仍不能超过快照每 case 的 200。

目的：若 nsys GPU gaps 明显减少且无 profiler token/s 提升，支持 host launch 是原因之一。若 FP8 仍远慢于 BF16，继续看 FP8 GEMM 和 quant。Graph 可能同时影响编译/融合，因此结论是“这套 graph/执行配置的效果”，要更严格隔离需继续核对 compilation config。

### 15.2 batch 1→4→8：验证小 M 效率

独立 offline 试验：

```bash
for p in fp8-local fp8-official bf16 awq; do
  for b in 1 4 8; do
    python "$OUT/profile_offline.py" \
      --profile "$p" --requests-file "$REQUESTS" --case w1024_o128 \
      --profiler none --batch-size "$b" --count 3 \
      --output "$OUT/evidence/${p}_batch${b}_no_profiler.json"
  done
done
```

目的：看扩大实际 decode token 行数后 FP8/BF16 相对差距如何变化。按各轮 `sum(output_tokens)/wall_seconds` 算 offline batch 产出；batch latency 与单请求 latency 一并记录。batch=8 的总吞吐提升不表示单用户时延更好，实际 M 也要 trace 验证。

24GB 卡、长上下文、大 batch 可能 OOM；记录失败并缩小范围，不改权重/显存比例后继续声称同一对照。当前 benchmark 的 concurrency 来自冻结 protocol，只改 `max_num_seqs=8` 不会把 HTTP 客户端并发从 1 自动变 8。HTTP 并发实验需要另建并记录有效协议，不能冒充原并发1结果。

### 15.3 prefill budget：验证 chunked prefill 的影响

```bash
for p in fp8-local fp8-official bf16 awq; do
  for budget in 1024 2048 4096; do
    python "$OUT/profile_offline.py" \
      --profile "$p" --requests-file "$REQUESTS" --case w4096_o256 \
      --profiler none --batch-tokens "$budget" --count 3 \
      --output "$OUT/evidence/${p}_prefillbudget${budget}.json"
  done
done
```

目的：改变一次 scheduler step 的 token budget，观察 prefill chunk 数、大 M GEMM 效率和 TTFT 的关系。不要把 budget 增大导致的 prefill 改善误归因于更换 FP8 后端。用输出1 + nsys 看 prefill 分块，再用完整输出评估 E2E。

### 15.4 普通 FP8 与 BLOCK FP8 的对照边界

这可以回答“慢的是 FP8 整体，还是当前 BLOCK scheme/后端”，但需要**单独正确导出的 checkpoint**和质量验证。不能把现有 block 模型的 config 改成 per-tensor，不能删除 scale，不强塞不兼容 loader。新方案的量化粒度、校准、backend、质量都是新变量，应作为下一阶段实验。

4090D 上开 `VLLM_USE_DEEP_GEMM=1` 或安装 FlashInfer，不能越过不满足的 SM 条件；优先核实这套 checkpoint 在 SM89 上可用的实现，不盲目照抄 Hopper 优化。

### 15.5 按逐请求记录计算 decode 近似，避免 p95 相减

```bash
export CASE=w1024_o128
python - <<'PY'
import json, os
from pathlib import Path
from test_server import percentile
root, case = Path(os.environ['EXP']), os.environ['CASE']
for profile in ('bf16','awq','fp8-official','fp8-local'):
    file = root/profile/case/'requests.jsonl'
    estimates = []
    for line in file.read_text().splitlines():
        e = json.loads(line)
        if e.get('phase') != 'measure' or not e.get('ok'):
            continue
        cm, usage = e['result']['client_metrics'], e['result']['usage']
        ttft, n = cm.get('time_to_first_text_seconds'), usage.get('completion_tokens', 0)
        if ttft is not None and n > 1:
            estimates.append((e['elapsed_seconds']-ttft)/(n-1))
    if estimates:
        print(profile, 'n=', len(estimates), 'decode_proxy_p50_ms=',
              1000*percentile(estimates,.5), 'decode_proxy_p95_ms=',
              1000*percentile(estimates,.95))
PY
```

目的：检查后续文本事件阶段是否稳定变慢。它仍含 stream/网络/排队/最后收尾，不称为 GPU TPOT。更细的 token latency 需要可靠逐 token 的引擎指标；SSE event 不保证一事件一 token。

### 15.6 正式复测与轮次波动

在没有 Nsight/torch profiler 的情况下：

```bash
python "$DEMO/scripts/benchmark.py" \
  --model fp8-local --model fp8-official --model bf16 --model awq \
  --case w1024_o128 --case w4096_o256 --requests 80 --rounds 3
```

这用于 baseline 多轮复测，不包含 graph 消融配置。再做一轮调整模型顺序或分次交错执行，观察时钟、温度、其它进程和运行次序影响。每轮分别报告，不能把多个精度合并；正式吞吐与 p95 仍按原脚本口径汇总。

原报告单轮 80 个请求可以说明本次现象；后续多轮不是追溯修改原结论。调参后另记录实验 ID、有效协议、请求子集 hash 和配置。

## 16. Step 13：写出可复查的性能结论

### 16.1 每个代表热点留一张证据表

| 字段 | 记录要求 |
|---|---|
| 实验身份 | run ID、snapshot/subset hash、代码 commit、配置 SHA |
| 设备与软件 | GPU UUID/CC、driver、CUDA build、torch/vLLM/Triton |
| 模型身份 | checkpoint/source revision、config/weight hash、loader |
| 请求与阶段 | case、request ID、prefill/decode、step/token 范围 |
| 算子 | 层、投影、逻辑 M/N/K、scale 粒度、实际 padded shape |
| kernel | 完整名、grid/block、registers/shared memory、duration |
| 指标 | DRAM read/write、带宽、Tensor、occupancy、waves |
| nsys | 所选窗口、launch 次数、GPU busy、gaps、量化占比 |
| 假设与验证 | 单变量改动、预期、实际变化、无 profiler 复测 |
| 证据路径 | nsys-rep、ncu-rep、日志、截图、JSON/CSV |

### 16.2 三种可以使用的结论措辞

**如果证实为小 M GEMM：**

> 在 GPU UUID …、SM89、vLLM … 的 block FP8 路径下，local/official 都命中 … kernel。decode M=1 的对应 … 投影比 BF16 多 … us；prefill M=… 则 …。扩大 batch 后差距 …，因此主要证据支持当前 block kernel 的小 M 效率限制。

**如果证实为动态量化开销：**

> FP8 GEMM 本身节省 … us，但每 step 的 group quant/scale 累计增加 … us。选中窗口总计 …，足以解释 decode 时间增加的 …%。Graph 消融后 …，动态量化仍是主要增量。

**如果证实为 host launch：**

> FP8 新增 … 次小 kernel；eager 的 GPU gaps 占 …。Graph 配置后 gaps 降至 …，无 profiler 吞吐由 … 到 …，支持 host 发射/执行路径开销显著。

空白必须由真实数据填写。不能因为源码预期 Triton 就提前写“根因已确认”。可能同时存在多个原因，分别量化它们能解释的比例。

### 16.3 每步节省与整体收益要能闭合

若测得 decode 每 token 增量约 `Δt`，则固定输出长度 O 的请求总增量可粗估为 `(O-1)*Δt`，再加 prefill/调度/采样的变化，与原 E2E 差异对照。

这只用于一致性检查；不能用不同请求的 p95、不同 profiler 时钟与 replay 耗时混算。对可优化部分占比 f、该部分加速 S，可用 Amdahl 估计整体上限：`1 / ((1-f)+f/S)`。若 quant 只占 5%，不能靠它解释全部 40% 吞吐损失。

## 17. 常见故障：症状 → 检查命令 → 处理

| 症状 | 检查 | 处理 |
|---|---|---|
| nsys 只有 HTTP/Python，没有 CUDA | 进程树、GPU worker PID、命令包裹位置 | 在模型服务/离线入口外启动 nsys，容器内执行 |
| capture 一直不开始 | `/start_profile` 返回码、profiler config、worker 日志 | CUDA profiler 必须配置并在 worker 触发 |
| start/stop 成功但窗口为空 | worker 注入、delay/max_iters、请求是否在 gate 内 | 先取消 iteration gate，抓一次完整正式请求验证 |
| NCU `No kernels were profiled` | regex、`--target-processes all`、CUDA start | 先用 nsys 确认名称，再减小过滤限制 |
| NCU 抓到加载/repack，不是推理 | `--profile-from-start off` 是否生效 | 保留 worker gate，预热在 start 之前 |
| `ERR_NVGPUCTRPERM` | 第 4.3 节权限与容器 capabilities | 管理员按 NVIDIA 文档赋权；不用反复重装 CUDA |
| CPU sampling 无权限 | `nsys status --environment` | 首轮保留 `--sample=none --cpuctxsw=none` |
| `Unknown metric/section/report` | 第 4.2 节导出的 help/list/query | 按实际版本选 ID；不盲搬其它架构名字 |
| NCU 运行特别慢 | metric 数、replay passes、过滤窗口 | basic + 1 个 kernel 起步，缩到 1～2 step |
| NCU 服务请求超时 | replay 放大时延、HTTP socket timeout | 优先离线自动请求；不能把 profiler 超时报正式失败 |
| NCU 卡住/NCCL hang | 是否 TP>1/多进程通信 kernel | 回到本项目 TP=1；跨 GPU 需专门处理 mandatory concurrency |
| nsys 报告丢失/未结束 | 服务是否仍在跑、工具日志是否完成导出 | 正常 Ctrl+C，等待 flush，不用 kill -9 |
| Graph 节点看不到 | 是否只收集 graph-level、shape 是否实际 graph | 用 node 诊断，确认 graph launch 和编译配置 |
| GPU 利用率很高但 token/s 低 | nsys 周期、ncu 小 kernel 指标 | utilization 是忙碌采样，不是有效算力/带宽比例 |
| FP8 显存没有减半 | KV cache 自动占据剩余预算、设备其它进程 | 分开看权重、KV/workspace、设备级采样；不能只比 nvidia-smi |
| local 与 official p95 不同但均值接近 | 逐请求、telemetry、多轮 | 排查尾部异常，不用单轮尾部直接推翻同 backend 假设 |
| 源码模块/符号不存在 | `vllm.__version__`、module `__file__` | 实际版本可能非 0.14.1；按安装源码重核对 |
| helper 输出文件已存在 | 路径与是否 application replay | 用唯一前缀；application replay 启用下面支持的唯一输出 |

### 17.1 NCU 必须用在同一服务时怎么办

可以让 NCU 启动整个服务进程树，再由另一终端发请求。首轮不推荐，因为 HTTP 超时和 replay 需要额外处理。

终端 A：

```bash
export PROFILE=fp8-local CASE=w1024_o128 EAGER=1
export KERNEL_RE='替换为实际目标kernel正则'
ncu --target-processes all --profile-from-start off \
  --kernel-name-base demangled --kernel-name "regex:$KERNEL_RE" \
  --launch-count 1 --set basic \
  --export "$OUT/ncu/local_http_one_kernel" \
  bash "$OUT/launch_server.sh"
```

终端 B 使用第 8.2 节回放工具，`--control --count 1 --timeout 1800`，换输出 JSON 路径。回放结束后正常退出服务，使 profiler 写完文件。这里 timeout 加长只是诊断容忍 replay，不改正式压测超时标准。

NCU 的 attach 模式通常用于已由 `ncu --mode launch` 注入启动的目标；不要把它当成任意运行中 Python PID 的通用无侵入 attach。对本项目最可靠的是由 NCU 启动目标进程树。

### 17.2 Docker 操作示例

先找正在运行的真实目标容器：

```bash
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}'
# 用上一步的真实 ID：
export TARGET_CID=实际容器ID
docker exec -it "$TARGET_CID" bash
```

进入后设置第 4 节变量，在容器内 `command -v nsys ncu`，执行主流程。若缺少计数器权限，`docker exec` 不能补上创建时缺失的 capability；由管理员使用同一个已验证镜像启动单独诊断容器，例如：

```bash
# IMAGE 取自原部署的真实镜像，不在实验中顺手换最新版。
export IMAGE=实际已验证的镜像名或摘要
export HOST_REPO=/实际宿主机路径/AI_system
export HOST_WEIGHTS=/实际宿主机路径/weights_pub
docker run --rm -it --gpus '"device=0"' \
  --cap-add=SYS_ADMIN \
  --mount "type=bind,src=$HOST_REPO,dst=/workspace/AI_system" \
  --mount "type=bind,src=$HOST_WEIGHTS,dst=/workspace/weights_pub,readonly" \
  --entrypoint bash "$IMAGE"
```

目的：使用独立诊断容器，权重只读挂载，输出保存在 repo 的 out 挂载内。需要原 venv、工具安装目录等时按真实部署补挂载；此模板没有假定 venv 已打进镜像。GPU device 编号按宿主机 UUID/部署修改。不要把主服务与诊断容器同时放在同一张卡上跑对照。

### 17.3 host 空洞明显时，再补 CPU sampling

确认 `nsys status --environment` 支持后，对独立小窗口将首轮的：

```text
--sample=none --cpuctxsw=none
```

改成该版本支持的 `--sample=process-tree --cpuctxsw=process-tree`，采样类型与权限以本地 help 为准。结合 CPU stack、OS runtime、线程状态区分 Python 执行、IPC 等待和 GPU 同步。某个 `futex` 长等待也可能是正常空闲 worker，只有与关键 GPU 空洞对齐才值得归因。

## 18. 报告带回 Windows 与最小交付清单

Linux 上先检查产物：

```bash
find "$OUT" -maxdepth 3 -type f \
  \( -name '*.nsys-rep' -o -name '*.ncu-rep' -o -name '*.json' -o -name '*.csv' \) -print
```

在本地 PowerShell 用实际 SSH 地址和目录拉回；以下是模板，不在 Linux 终端执行：

```powershell
$profileDir = 'D:\workspace\learing\AI_system\out\quantization_profiling\nsight-results'
New-Item -ItemType Directory -Force -Path $profileDir
scp -r 'user@gpu-host:/workspace/AI_system/out/quantization_profiling/nsight_实际时间/*' $profileDir
```

GUI 用 File → Open 打开：

- Systems：`*.nsys-rep`，查看时间线和选中窗口统计。
- Compute：`*.ncu-rep`，查看 Details、Source/SASS 和 baseline comparison。
- JSON/CSV：检查请求、环境、配置与整理出的证据。

GUI 版本应能读取采集工具版本，优先使用相同或兼容版本；不要求 Windows 本机 GPU 与目标 4090D 相同。不要提交大型 trace/权重进 Git，提交文档和必要小型证据表即可。

第一轮至少留下：

1. 四模型 `w1024_o128` 的 nsys 时间线与相同阶段对照截图。
2. local/official 的一个 decode GEMM、一个 prefill GEMM、一个 quant 热点 NCU 报告；BF16/AWQ 的对应 GEMM。
3. 每个热点的 M/N/K、kernel/launch 配置、耗时、DRAM、Tensor、occupancy 证据表。
4. 一次 eager/graph 或 batch 消融，以及无 profiler 复测结果。
5. 尚未证实的假设和下一步，不写成既定根因。

## 19. 本文使用的官方资料

优先按已安装版本的 help 使用命令；以下链接用于核对机制与版本差异：

- [vLLM 0.14.1：Profiling vLLM](https://docs.vllm.ai/en/v0.14.1/contributing/profiling/)：服务/离线采集思路与多进程注意事项。
- [vLLM 0.14.1：ProfilerConfig](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/config/profiler.py)：cuda/torch、delay/max iterations 配置。
- [vLLM 0.14.1：worker profiler wrapper](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/profiler/wrapper.py)：worker CUDA start/stop 与 iteration gate。
- [vLLM 0.14.1：LLM 接口](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/entrypoints/llm.py)：离线 start_profile / stop_profile。
- [vLLM 0.14.1：block FP8 分派源码](https://github.com/vllm-project/vllm/blob/v0.14.1/vllm/model_executor/layers/quantization/utils/fp8_utils.py)：架构条件与实际后端；目标机安装源码应另外保存。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：capture、CUDA Graph、NVTX、CLI 选项。
- [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)：统计和结果解读。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：过滤、section、metrics、导入导出与 replay 参数。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：replay、cache/clock、graph、指标限制。
- [NVIDIA ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)：性能计数器权限与容器要求。

本文给出的是针对这个 demo 的诊断命令与判断标准；没有把远端未执行的采集、质量回归或优化结果写成已完成。

## 20. 文档本地验证记录

本次已对文档中的 Bash 代码块执行 `bash -n`，对 Python 示例执行 AST 语法检查；并用 mock 验证四模型 loader 参数、冻结请求 ID/长度、预热先于 profiler、输出1仍保留原 engine 长度、batch/无 profiler 路径、application replay 唯一输出、异常时停止 profiler、HTTP 控制时序。验证结果位于仓库 `out/nsight-doc-validation/validation.json`。

这些检查不加载 GPU 权重，不能证明目标驱动/Nsight 的兼容性、实际 kernel 名称和计数器可采；真正的 Linux 4090D 采集需按本文执行。
