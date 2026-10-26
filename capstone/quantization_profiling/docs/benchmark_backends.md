# benchmark：只选 FP8、强制 Marlin 与控制 eager

以下在 Linux GPU 目标机的仓库根目录执行，激活统一环境 `/workspace/venvs/vllm-0112`（实际 vLLM 为 0.14.1）。不需要重新生成冻结请求；模型、负载和请求子集选择仍沿用原流程。

## 1. 只测试某个模型

```bash
# 官方 FP8；fp8-origin 与 fp8-official 是同一个预设。
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-origin --requests 80

# 本地 FP8。
python capstone/quantization_profiling/scripts/benchmark.py --model fp8-local --requests 80

# 只比较这两个，不运行 BF16 / AWQ。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --model fp8-origin --requests 80
```

省略 `--case` 测五组负载；指定 `--case w1024_o128` 只测一组。`--model` 可重复；选择同一模型的两个别名会因重复选择被拒绝。

benchmark 默认轻量预检模型配置、架构、文件存在性与 backend 请求，跳过完整权重 SHA256 扫描；`--no-verify-weights` 可显式选择这一默认行为。需要冻结权重身份核验时追加 `--verify-weights`，例如：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --case w1024_o128 --requests 80 --verify-weights
```

开关不改变 eager、backend、请求或临时目录策略。快照哈希仍校验；启动成功仅证明运行路径可用，不能替代完整权重身份验证。质量对比的直接 verify_model() 调用仍默认严格。

## 2. 控制 enforce_eager

| benchmark 参数 | 本次 enforce_eager | 含义 |
|---|---|---|
| 不传 | 沿用各模型配置 | 当前四个默认配置都是 true |
| `--enforce-eager` | true | 禁用 CUDA Graph |
| `--enforce-eager true` | true | 同上；也接受 1 |
| `--enforce-eager false` | false | 允许 vLLM 按自身配置使用 Graph；也接受 0 |
| `--no-enforce-eager` | false | 与上面 false 等价 |

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --enforce-eager false --case w1024_o128 --requests 80

python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --enforce-eager true --case w1024_o128 --requests 80
```

这些是本次内存中的配置覆盖，不改 `configs/*.yaml`、冻结快照或原始请求。`--config` 自定义文件也接受相同覆盖。`false` 允许 Graph，不保证每种 shape 都实际进入 Graph；由启动日志和 nsys 确认。

`run_model.py` 的原有 eager 布尔参数仍用 `--enforce-eager` / `--no-enforce-eager`，上述带 true/false 值的写法是 benchmark 的新接口。

## 3. 官方 FP8：指定 Marlin

```bash
# 先检查计划：不读完整权重、不启动 GPU 服务；模型格式/初始化仍待实际运行核验。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-origin --fp8-backend marlin --enforce-eager false \
  --case w1024_o128 --requests 80 --rounds 1 --plan-only

# 实际运行：删除 --plan-only。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-origin --fp8-backend marlin --enforce-eager false \
  --case w1024_o128 --requests 80 --rounds 1

# 保持 eager，再测一次以拆开 backend 与 Graph 的影响。
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-origin --fp8-backend marlin --enforce-eager true \
  --case w1024_o128 --requests 80 --rounds 1
```

删除 `--case` 就测试全部五组。正式请求数不能超过快照中每组可用记录数。

实现是在服务子进程设置 `VLLM_TEST_FORCE_FP8_MARLIN=1`，通过 vLLM 0.14.1 的原生 `Fp8LinearMethod` 分派；量化 loader 仍是 `fp8`，不存在项目自行发明的 `--quantization fp8_marlin`。权重在加载阶段重排为 Marlin 格式，原始 checkpoint 不修改。

**该路线是 FP8 权重 + BF16/FP16 激活的 W8A16。** 它不执行原来的动态 FP8 激活量化，因此性能/数值语义与默认 W8A8 不同。`--dtype bfloat16` 在本项目保持原来的 BF16 非量化计算。质量需要单独回归，不能只凭速度认定更好。

源码依据：[vLLM 0.14.1 原生 FP8 分派](https://raw.githubusercontent.com/vllm-project/vllm/v0.14.1/vllm/model_executor/layers/quantization/fp8.py)、[Marlin FP8 权重准备与计算](https://docs.vllm.ai/en/v0.14.1/api/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8/)。该显式选项限定已核对的 vLLM 0.14.1，旧 0.11.0 环境会拒绝运行。

## 4. 当前 fp8-local 为什么不能直接强制 Marlin

| checkpoint 实际格式 | `--fp8-backend marlin` |
|---|---|
| 官方原生 `quant_method=fp8`，block128 | 支持请求 Marlin W8A16，需目标机验证初始化与 kernel |
| 本地 checkpoint 本身也是合法原生 `fp8` | 同样支持 |
| 当前本地 LLM Compressor 的 compressed-tensors FP8_BLOCK / W8A8 | 此版本不能仅靠 force-Marlin 变量切换，明确拒绝 |

vLLM 0.14.1 的 `CompressedTensorsW8A8Fp8` 不读取这个变量，其已有 `CompressedTensorsW8A16Fp8` 也不是把当前 BLOCK W8A8 直接切过去的通用入口。[W8A8 scheme 源码](https://docs.vllm.ai/en/v0.14.1/api/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8/)、[compressed-tensors 分派](https://docs.vllm.ai/en/v0.14.1/api/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors/)。

因此，对当前模型执行：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --fp8-backend marlin --requests 80
```

会在读取模型 config 的 precheck 阶段失败，写出失败原因；不会默默跑 Triton 并记为 Marlin。`--plan-only` 仍可能列出该请求计划，因为这个模式本身不保证模型格式已经验收。

若要把当前 local 也做 Marlin 对照，需要另外制作、验证原生 FP8 兼容的转换产物（tensor 名称、scale 语义/shape、index、忽略层、来源指纹等），或实现并验证版本固定的自定义运行方案。本次没有自动修改/重标注当前 checkpoint，也没有声称完成这类转换。

当前本地模型继续用：

```bash
python capstone/quantization_profiling/scripts/benchmark.py \
  --model fp8-local --fp8-backend auto --enforce-eager false \
  --case w1024_o128 --requests 80
```

## 5. auto、环境变量与记录

- `--fp8-backend auto`：让 checkpoint/平台决定默认分派；在 4090D 的当前 BLOCK FP8 路径通常为 Triton。**auto 不承诺强制 Triton**，换 GPU/版本时可能是其它 backend。
- auto 会在服务子进程设置 `VLLM_TEST_FORCE_FP8_MARLIN=0`，覆盖终端中残留的 force-Marlin；请用新的显式参数请求 Marlin。
- 每个服务有独立环境副本，不修改 benchmark 父进程，不把某个模型的 Marlin 设置泄漏给下一模型。
- Marlin 请求遇到 `VLLM_BATCH_INVARIANT` 开启或 `VLLM_MARLIN_INPUT_DTYPE` 设置时会报冲突；先核对并 unset 这些与 W8A16 路线冲突的变量。
- `--fp8-backend` 命令行覆盖只作用于所选 FP8；只选 BF16/AWQ 却要求 marlin 会报错。AWQ 的 `awq_marlin` 与这个 FP8 开关独立。

记录位置：

| 产物 | 内容 |
|---|---|
| `--plan-only` JSON | 模型、eager、backend 请求、实际命令、受控环境覆盖 |
| `run.json` → models[].config | 本次生效的 `enforce_eager` 与 `fp8_backend` |
| 各组 `summary.json` | eager/backend 请求、环境覆盖、初始化核验、`weight_verification` |
| `run.json` → models[].weight_verification / models[].model.weight_verification | 状态 skipped/passed，区分默认轻量检查与本次完整哈希通过；失败另读 mode/error |
| 实验根 `events.jsonl` | 模型检查、显式哈希进度和服务启动/health 子阶段 |
| 各组 `worker_events/` | `--nvtx` 开启时的实际 worker 初始化、加载、prefill/decode 日志 |
| `server.log` | 入口请求信息、vLLM 实际 dtype/loader、Marlin 初始化信息 |
| `Week1_Report.md` | eager/backend 设置表；Marlin 标为 W8A16 |

`weight_verification.status=skipped` 不是模型预检失败，也不因 case passed 自动变成 passed；显式全量验证通过才记录 passed。Marlin 请求必须在服务启动日志中出现其权重准备阶段的信息，缺失时该组失败。`backend_initialization_verified=true` 只证明初始化证据；`cuda_kernel_verified` 仍为 false，实际 kernel 用 Nsight 确认。

vLLM 的 Marlin 初始化 warning 可能写着 GPU 没有原生 FP8 支持；显式强制时也会使用这条通用文案，不能据此推断 4090D 不支持 FP8。

## 6. 建议先做的对照矩阵

先选 `w1024_o128`，使用相同 80 条正式请求、5 次预热：

| 模型 | backend | eager | 目的 |
|---|---|---|---|
| fp8-origin | auto | true | 原路线参照 |
| fp8-origin | marlin | true | 只改 FP8 执行路线 |
| fp8-origin | marlin | false | 再看允许 Graph 的效果 |
| fp8-local | auto | true / false | 本地模型的 eager/Graph 对照 |

之后扩展长输入和五组负载，再加多轮。所有速度在无 profiler 的 benchmark 测，kernel 验证另用 [Nsight 手册](nsight_fp8_performance_analysis.md)。

本次开关与环境传递有离线单元测试；未在远端 4090D 上实际执行 Marlin 性能测量。
