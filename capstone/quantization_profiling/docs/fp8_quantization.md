# 使用 LLM Compressor 将 Qwen3-8B 转为分块 FP8

更新：2026-10-21。入口为 `scripts/quantize_fp8.py`。使用 LLM Compressor 真实量化并保存 FP8 权重，不是仅修改 config，也不是直接对 BF16 张量调用 `.to(float8)`。

## 1. 与 Qwen 官方配置的对应关系

目标是对齐官方公开的数值方案：

```json
{
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]
}
```

本脚本使用 `FP8_BLOCK`，不是 `FP8_DYNAMIC`：

| 项目 | 本脚本 |
| --- | --- |
| 模型 | 本地 dense Qwen3，默认 `/workspace/weights_pub/Qwen3-8B` |
| 加载精度 | BF16 |
| 权重 | E4M3 FP8，静态 128×128 分块、对称量化 |
| 激活 | 推理时动态量化，每 token、每 128 个输入通道一组 |
| 量化方法 | QuantizationModifier 的 RTN（舍入），无训练、无任务校准集 |
| 目标层 | Linear；排除 lm_head；embedding、norm 等保留 BF16 |
| 导出 | compressed-tensors，真实 `quant_method="compressed-tensors"` |
| 输出路径 | `/workspace/weights_pub/Qwen3-8B-FP8-local` |

官方原生权重使用 `weight_scale_inv` 等元数据；本工具使用 `weight_scale` 和 compressed-tensors 配置。两者数值方案接近，文件元数据不同。**不要把导出的 quant_method 手动改成 fp8，也不要擅自重命名 scale 或取倒数。** vLLM 0.11.0 已有 compressed-tensors BLOCK-FP8 加载路径，启动时让它自动识别。

scale 实际 dtype 以文件为准，BF16 加载时工具可能生成 BF16 scale，不承诺与官方 FP32 scale 或舍入结果一致。官方公开的分块配置不足以保证逐位复现。质量需配对评测后判断。

## 2. 独立量化环境

你已跑通的推理环境保持 `torch 2.8.0+cu129 / Transformers 4.57.6 / vLLM 0.11.0`。下面的依赖只安装到独立量化环境。

本脚本对应的固定版本为：

- LLM Compressor 0.8.1。
- compressed-tensors 0.12.2。
- PyTorch **2.8.0+cpu**，用于 CPU 离线转换，不下载 CUDA 运行库。
- Transformers 4.56.2、Accelerate 1.10.1。

原因是 LLM Compressor 0.8.1 的 release 依赖声明要求 Transformers ≤4.56.2、Accelerate ≤1.10.1。不能强制安装到原推理环境，不能使用 `--no-deps` 忽略冲突。本项目原 requirements.txt 不作修改。

在 Linux 主机，从仓库根目录运行；Python 3.10/3.11/3.12 均可按依赖解析结果选择：

```bash
python3 -m venv /workspace/venvs/qwen-fp8-quant
source /workspace/venvs/qwen-fp8-quant/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r capstone/quantization_profiling/requirements-quantization.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

预期分别输出 `2.8.0+cpu`、`None`、`False`。即使机器安装了 4090D 和 NVIDIA 驱动，CPU 版环境也应显示 CUDA 不可用。这不影响量化产物随后在另一个 GPU 环境中运行。

requirements-quantization.txt 已改为固定 `torch==2.8.0+cpu`，并配置上海交大 PyTorch CPU 镜像作为附加源；普通依赖可使用清华镜像。**不能只把版本改成 `torch==2.8.0` 并仅指定清华源，就认定下载的是 CPU 版**：Linux 上普通 PyPI 构建可能依赖 CUDA 库。显式 `+cpu` 可以避免误选。

如果已经在独立量化环境装了旧的 CUDA 版 torch，激活该环境后重新执行上述 requirements 安装命令即可切换；旧 CUDA 依赖包可能仍保留，但不会被 CPU 版 torch 使用。不要在 `(cutedsl-venv)` 中执行这次切换。

量化环境不需要安装 vLLM、torchvision 或 torchaudio；完成转换后回到原来的 GPU 推理环境加载。CPU wheel 使用单独的 [上海交大 PyTorch 镜像](https://mirror.sjtu.edu.cn/docs/pytorch-wheels)，普通包使用清华源：

```text
CPU torch: https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
其它依赖:  https://pypi.tuna.tsinghua.edu.cn/simple
```

已核对该镜像包含 `torch-2.8.0+cpu` 的 Python 3.10/Linux x86_64 wheel，并通过小范围 HTTP 读取确认返回真实 wheel 数据；实际下载速度取决于网络。若镜像不可用，将 requirements-quantization.txt 中的附加源一行替换为官方 CPU 源后重试：

```text
--extra-index-url https://download.pytorch.org/whl/cpu
```

也可以先单独安装 CPU torch，再执行 requirements 命令补其它依赖：

```bash
python -m pip install "torch==2.8.0+cpu" \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
```

切勿在 Linux 上仅使用清华 PyPI 源安装不带 `+cpu` 的 torch，然后认定它是 CPU 构建。

CPU 构建和下载方式以 [PyTorch 2.8.0 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280) 与 [CPU wheel 索引](https://download.pytorch.org/whl/cpu/torch/) 为依据。

## 3. 预检与正式转换

先预检，不导入 torch/Transformers/Compressor，不加载模型，不写输出：

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py --dry-run
```

它检查本地 config、分片索引和 safetensors header、所有 Attention/MLP 投影、原始浮点权重和 tokenizer，并打印方案、预计源权重字节数、当前依赖版本。预检通过不表示依赖安装正确或目标机转换已成功。

**推荐先使用 CPU 完成 RTN 转换：**

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local \
  --device cpu
```

默认就是 CPU。这一步只做权重统计、缩放、FP8 cast 和保存，不执行 FP8 模型前向，因此不需要 CPU 支持原生 FP8 MMA。需要足够的主机内存容纳源 BF16 模型、临时张量和保存过程副本；8.2B BF16 权重约 15GiB，实际峰值会更高。CPU 转换速度与主机内存占用须在你的目标机验证。

如果希望使用 GPU，必须另建安装了 **CUDA 版 PyTorch** 的量化环境。上面默认的 CPU 版环境不能使用 `--device cuda:0`，不要把此命令直接接在 CPU 安装流程后执行。

在已有独立 CUDA 量化环境（LLM Compressor 0.8.1、compressed-tensors 0.12.2、Transformers 4.56.2、Accelerate 1.10.1、torch 2.8.0+cu129）中，才可单独尝试：

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local-gpu \
  --device cuda:0
```

GPU 转换前应停止占用显存的模型服务。约 20GiB 空闲显存不保证容纳转换峰值；原 BF16 模型、观察器临时张量与保存副本都会占空间。GPU 路线记录 PyTorch allocated/reserved 的采样统计峰值。OOM 后保留失败产物，改用 CPU 或释放资源，并选择新的输出目录。

可选参数：

| 参数 | 默认值 / 用途 |
| --- | --- |
| --model | `/workspace/weights_pub/Qwen3-8B`，只读本地目录 |
| --output | `/workspace/weights_pub/Qwen3-8B-FP8-local`，必须尚不存在 |
| --device | cpu；也可 cuda、cuda:0 等，cuda 等价 cuda:0 |
| --max-shard-size | 2GB，限制导出分片大小，不限制运行内存峰值 |
| --seed | 42 |
| --source-revision | unknown；可填已确认的不可变源 commit，脚本不代查来源 |
| --log-dir | 项目的 log/ |
| --dry-run | 只检查源文件并打印计划 |
| --verify-only | 复核已有成功产物的格式、全部目标层、scale 和 hash |
| --debug | 失败时打印 traceback |

源和输出不能相同或相互包含；任何已存在的输出目录都拒绝覆盖。脚本不下载模型、不修改原始权重、不使用现有 100 条质量任务做校准。

## 4. 导出后会检查什么

不以 config 存在作为成功标准。脚本逐层检查：

1. config 确实是 compressed-tensors，状态为 compressed。
2. 权重方案为 FP8 BLOCK [128,128]，激活方案为动态 GROUP 128。
3. Qwen3 每层七个投影（q/k/v/o、gate/up/down）全部保存为 E4M3 FP8；36 层模型应有 252 个量化 Linear 权重。
4. 每个投影有 weight_scale，shape 为 `[ceil(out/128), ceil(in/128)]`，且所有 scale 都有限、严格大于零。
5. 非目标权重仍为 BF16；原模型张量形状保持一致。
6. tokenizer 已保存，重新加载后的 chat template 与源模板 hash 一致。
7. 记录源与输出配置、tokenizer、权重等文件的完整 SHA256。

hash 会顺序读取大型分片，耗时属于正常现象。请保持源目录静止，避免与下载或其它写入同时运行。

产物包括：

```text
/workspace/weights_pub/Qwen3-8B-FP8-local/
├── config.json
├── model*.safetensors
├── model.safetensors.index.json   # 分片保存时生成
├── tokenizer.json
├── tokenizer_config.json
├── ...                           # 模板、生成配置等
└── quantization_manifest.json
```

manifest 记录完整 recipe、源权重与产物 hash、原始形状、依赖版本、源 revision 标签、模板 hash、阶段耗时、验证结果、GPU 峰值（GPU 路线）和完成状态。只有完整流程通过才标记 `complete`；失败/中断会标记 failed/interrupted，保留文件供排查，不自动删除或续传。

转换脚本日志写入项目 `log/quantize_fp8_<ID>.log`；LLM Compressor 自身日志目录为 `log/compressor_<ID>/`。

之后可独立复核（不需要 torch/Compressor，也不需要原模型目录）：

```bash
python capstone/quantization_profiling/scripts/quantize_fp8.py \
  --output /workspace/weights_pub/Qwen3-8B-FP8-local --verify-only
```

这验证的是静态模型产物，不验证回答质量，也不等于目标 GPU 推理已通过。

## 5. 回到原推理环境验证 vLLM

转换进程结束后，激活你原先的 vLLM 环境，然后执行：

```bash
python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm \
  --model /workspace/weights_pub/Qwen3-8B-FP8-local \
  --dtype bfloat16 --thinking off \
  --prompt '请用三句话解释 KV cache。' \
  --max-model-len 2048 --max-new-tokens 128
```

保持项目默认 quantization=auto，不显式传 `--quantization fp8`；此处应由保存的 compressed-tensors 配置选择后端。`dtype bfloat16` 控制未量化部分及相关输出浮点精度，不会将 FP8 权重改回 BF16。

服务方式：

```bash
python capstone/quantization_profiling/scripts/run_model.py \
  --serve --backend vllm \
  --model /workspace/weights_pub/Qwen3-8B-FP8-local \
  --dtype bfloat16 --host 0.0.0.0 --port 7081
```

检查启动日志，确认 compressed-tensors 和引擎实际 dtype/量化方法；具体 GPU kernel 需要 Profiler 证据。随后复用 test_server.py 做连通性/指标检查。`run_model.py --model qwen3-8b-fp8-local` 是相同本地目录的别名。

benchmark 现已支持本地产物，使用原推理环境和同一份 BF16 tokenizer 冻结请求：

```bash
# 五组负载，每组每轮5次预热、50次正式请求，默认1轮。
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-local

# 轮数可选，先查看计划再运行三模型对照。
python capstone/quantization_profiling/scripts/benchmark.py --all-models --rounds 2 --plan-only
python capstone/quantization_profiling/scripts/benchmark.py --all-models --rounds 2
```

旧 100 请求快照可以直接复用，默认按固定顺序取前 50 个正式请求，各模型使用相同子集。请求数还可用 `--requests N` 修改，但不能超过快照可用记录数。BF16/AWQ/本地 FP8 分别启动、停止并留存日志，汇总时按模型和负载分组；详见 [完整执行说明](week1_execution.md)。本次实现尚未在目标 4090D 上完成三模型真实对照。

模型能回答后，再对同一留出集做 BF16/官方 FP8/本地 FP8 三组评测，比较质量及独立的性能记录。官方自定义脚本与此配方的 scale 精度、舍入、保留层可能不同，不能提前宣称相同质量或速度。

## 6. 实现与验证边界

本地验证使用标准库测试 safetensors 结构、路径/覆盖保护、完整层清单、scale 值和错误留档；CPU 默认依赖、镜像索引和 wheel 小范围数据已核对，未完整下载或安装 torch；当前本地没有 torch、LLM Compressor 或目标 4090D，尚未进行真实 8B 转换及 GPU 加载。这里的固定版本来自官方 release 依赖与代码核对，目标机仍需 pip check、实际转换和推理验收。

参考固定版本资料：

- [LLM Compressor 0.8.1 的 Qwen3 FP8_BLOCK 示例](https://github.com/vllm-project/llm-compressor/blob/0.8.1/examples/quantization_w8a8_fp8/fp8_block_example.py)
- [0.8.1 依赖约束](https://github.com/vllm-project/llm-compressor/blob/0.8.1/setup.py)
- [compressed-tensors 0.12.2 的 FP8_BLOCK 定义](https://github.com/vllm-project/compressed-tensors/blob/0.12.2/src/compressed_tensors/quantization/quant_scheme.py)
- [vLLM 0.11.0 的 compressed-tensors FP8 执行实现](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py)
- [Qwen 官方 FP8 配置](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json)
