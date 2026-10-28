# 非 server NSYS / NCU 脚本参数说明

两个入口都通过 profile_kernels.py 调用 LLM.generate。模型由参数选择，不再保留按 FP8/BF16 命名的独立包装脚本。运行前的环境、冻结快照和产物解释见 [离线采集流程](offline_nsight.md)。

推荐使用命名参数；旧位置参数仍然可用。显式命令行参数优先于相应环境变量，环境变量优先于脚本默认值。命名模型/模式等参数与对应的位置参数不要混用。

## 1. 支持的模型

| --model | MODEL_ROOT 下的目录 | 含义 |
|---|---|---|
| bf16 | Qwen3-8B | BF16 基线 |
| awq | Qwen3-8B-AWQ | AWQ checkpoint，默认匹配实际 Marlin 热点 |
| fp8-official | Qwen3-8B-FP8 | 官方 FP8 checkpoint |
| fp8-local | Qwen3-8B-FP8-local | 本地 FP8 checkpoint |
| all | 上述四项 | 仅 NSYS 支持，串行采四种模型 |

--model 选择项目已定义的模型 profile，不是任意模型路径。权重根目录通过 MODEL_ROOT 指定；程序验证对应模型的量化格式。NCU 每次指定一个模型，便于解释所选 kernel。

## 2. profile_offline_nsys.sh

### 调用格式

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh \
  --model fp8-local --mode graph --scope steady

# 等价的位置参数写法：
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh fp8-local graph steady

# 查看帮助，不加载模型：
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh --help
~~~

### 全部命令行参数

| 参数 | 默认值 | 具体作用 |
|---|---|---|
| --model MODEL | 必填 | 选择一个模型，或 all |
| --mode eager / graph / all | graph | eager 设置 enforce_eager=true；graph 允许编译与 Graph；all 分别运行两种模式 |
| --scope steady / all | steady | steady 只记录预热后的正式请求；all 从程序启动记录到结束 |
| --case ID | PROFILE_CASE 或 w128_o128 | 选择冻结请求 case；输入/输出长度来自该 case 的冻结数据 |
| --count N | PROFILE_COUNT 或 1 | 正式 LLM.generate 的调用次数，不含预热 |
| --batch-size N | PROFILE_BATCH_SIZE 或 1 | 每次 generate 同时提交的请求数，也设置引擎 max_num_seqs |
| --batch-tokens N | PROFILE_BATCH_TOKENS 或 2048 | 引擎 max_num_batched_tokens，调度 token budget，不是固定生成长度 |
| --warmup N | 5 | 完整预热 generate 次数，必须至少 5；两种 scope 都实际运行预热 |
| --requests-file PATH | REQUESTS 或项目 data/frozen/requests.jsonl | 冻结请求文件，同目录 manifest 等文件也必须存在并通过校验 |
| --output-dir PATH | NSYS_OUT 或项目 log/nsys_offline | 报告、SQLite、覆盖检查、日志和 workload JSON 输出目录 |
| -h / --help | — | 显示帮助并退出 |

--count 与 --batch-size 相乘，才是正式请求总数；例如 count=3、batch-size=2 表示三次 generate、共六个请求。冻结快照必须有足够数量的正式请求。预热为 N 次 generate，每次同样提交 batch-size 个请求。

三个 all 的含义独立：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh \
  --model all --mode all --scope all
~~~

该命令产生 4 模型 × 2 模式，共 8 份从初始化开始记录的报告。若只想八份稳态报告，用 --scope steady。

### steady 和 all 的边界

| scope | 模型加载、profile_memory、KV cache、编译/捕获 | 五次或更多预热 | 正式请求 | CUDA profiler gate |
|---|---|---|---|---|
| steady | 执行但不记录 | 执行但不记录 | 记录全部正式 generate | 预热后 start_profile，全部正式请求结束后 stop_profile |
| all | 执行并记录 | 执行并记录 | 执行并记录 | 不启用，NSYS 从进程启动采集 |

两个窗口都开启控制进程和真实 worker 的 NVTX。NSYS 固定记录 CUDA/NVTX/OSRT、Graph node，关闭 CPU sampling/context switch，结束后自动导出 SQLite 并检查测量范围和 Graph 执行证据。all 的全局 Summary 包含启动与预热，应限定 offline/measure 才分析正式请求。

## 3. profile_offline_ncu.sh

### 调用格式

~~~bash
# 精确名称：这里是当前 FP8 Triton kernel 的名字。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --level detail \
  --kernel-name '_w8a8_triton_block_scaled_mm'

# 正则：匹配完整 demangled 名称中包含命名空间的 BF16 GEMV。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model bf16 --phase decode --level basic \
  --kernel-regex 'internal::gemvx::kernel<'

# 项目已核对的投影预设仍可使用：
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --target down --level detail

# 对应的旧位置参数：
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh fp8-local decode down detail

bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh --help
~~~

### 模型、阶段与粒度

| 参数 | 默认值 | 作用 |
|---|---|---|
| --model MODEL | 必填 | bf16、awq、fp8-official、fp8-local；每次一个模型 |
| --phase prefill / decode | decode | 同时选择 worker NVTX 阶段和该阶段的默认 profiler 窗口 |
| --level basic / detail | basic | 选择下面表格中的 section 集合 |
| --mode graph / eager | NCU_GRAPH=1 对应 graph，否则 0 对应 eager | 指定引擎执行模式；显式 --mode 覆盖合法的环境默认值 |

### kernel 选择：以下三项互斥

| 参数 | 默认值 | 匹配方式 |
|---|---|---|
| --target PRESET | 没有指定任何选择器时为 first4 | 使用已有矩阵或量化热点预设 |
| --kernel-name NAME | 无 | 精确匹配；符号含空格、括号、模板时必须整体加引号 |
| --kernel-regex REGEX | 无 | 正则部分匹配；不需要写 regex: 前缀，脚本自动添加 |

例如 --kernel-name foo 只匹配名称恰好为 foo 的 kernel；--kernel-regex foo 可匹配 foo、fooBar 等。不要把 NCU 原生的 regex: 前缀放进 --kernel-name；请使用 --kernel-regex。

| --kernel-name-base | 作用 | 适用例子 |
|---|---|---|
| demangled，默认 | 含命名空间、模板及参数的可读名称 | C++ 完整签名或 namespace 正则 |
| function | 去掉模板和参数的函数名称 | 只知道函数名时；需注意重名 |
| mangled | 编译后的修饰符号 | 从实际报告或二进制取得 mangled 符号 |

--target 预设使用 demangled；若要用其他 name base，请使用自定义名称或正则。Triton 的可读名字通常较短，直接传精确名称即可；C++ kernel 的精确匹配需要包含该 name base 下完整的名字。

### 调用选择与 worker 窗口

| 参数 | 默认值 | 作用 |
|---|---|---|
| --launch-skip N | 预设值，自定义 kernel 为 0 | 跳过 N 次满足 kernel 与阶段过滤条件的 launch |
| --launch-count N | first4 为 4，其余为 1 | 采集的匹配 launch 数；必须大于 0 |
| --delay-iters N | prefill=0；decode=8 | 正式请求 start_profile 后，worker profiler 的延迟迭代设置 |
| --max-iters N | prefill=1；decode=2 | worker 采集窗口长度；0 表示不自动按迭代数停止，由请求结束后的 stop_profile 停止 |

launch-count 不是 token 数、请求数、模型层数，也不是 NCU 为采计数器而 replay 的 pass 数。launch-skip 只跳过匹配项，不承担跳过初始化和预热的任务。

脚本会在实际 GPU worker 中启用 ProfilerStart/Stop，并增加对应阶段的 NVTX 过滤：prefill 选择 worker/prefill，decode 选择 worker/decode。这是 kernel replay 的筛选条件，不是把 replay mode 改成 range。独立 worker/sampling 不属于这两种阶段范围；选择 sampling kernel 不能期待在 prefill/decode 过滤下命中。

对原 w128_o128、batch=1，delay=8/max=2 对应正式请求第 8、9 次模型迭代，在第 10 次之前停止。iteration 是引擎迭代，不是直接传入的 token 下标。长输入的 chunked prefill 或不同 batch 会改变阶段边界，不能直接沿用旧窗口推断。

如果自定义 case 的 decode 尚未进入默认窗口，可使用更大的 delay/max；也可以显式采完整正式请求，再由阶段 NVTX 过滤选择实际 decode：

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --level basic \
  --case w4096_o128 --kernel-name '_w8a8_triton_block_scaled_mm' \
  --delay-iters 0 --max-iters 0 --launch-count 1
~~~

该 case 必须已存在于冻结快照。prefill 的 max=1 默认只包含一个模型迭代；如果输入被切成多个 chunk、希望覆盖所有 prefill 迭代，同样可用 delay=0/max=0，阶段过滤仍只选择 prefill。

### 原有热点预设

| --target | 匹配目标 | 默认 skip | 默认 count |
|---|---|---:|---:|
| first4 | 当前模型的矩阵 kernel 家族 | 0 | 4 |
| qkv | 第一层 QKV，按匹配调用顺序定位 | 0 | 1 |
| o | 第一层输出投影 | 1 | 1 |
| gate_up | 第一层合并 gate/up 投影 | 2 | 1 |
| down | 第一层 down 投影 | 3 | 1 |
| quant | FP8 独立分组量化 kernel | 0 | 1 |
| norm_quant | FP8 RMSNorm/量化融合 kernel | 0 | 1 |

前五个投影位置预设只允许原 w128_o128、batch=1，避免把历史调用顺序套到其他负载；quant/norm_quant 只允许 FP8 模型。显式 launch-skip/count 可以覆盖预设值。

自定义 kernel 选择不受上述投影映射限制，可用于其他冻结 case 和 batch，但用户需要核对所采阶段和 shape。一个名字可能覆盖多个投影：当前 FP8 四个线性投影都叫 _w8a8_triton_block_scaled_mm；名称只能筛家族，不能独自识别 down_proj。

### 不同粒度保留的具体 section

| level | section | 主要查看内容 |
|---|---|---|
| basic、detail | LaunchStats | grid/block、寄存器、共享内存、waves 等启动信息 |
| basic、detail | SpeedOfLight | 计算与内存吞吐概览 |
| basic、detail | Occupancy | 理论/实际占用与资源约束 |
| detail 额外增加 | MemoryWorkloadAnalysis | DRAM、L2、访存事务等 |
| detail 额外增加 | ComputeWorkloadAnalysis | 运算和执行管线 |
| detail 额外增加 | SchedulerStats | eligible/issued warps、调度情况 |
| detail 额外增加 | WarpStateStats | warp 等待状态 |
| detail 额外增加 | SourceCounters | 可用的源码/指令级统计 |

因此 basic 为 3 个 section，detail 为 8 个。先 basic 确认目标、再 detail 分析少量调用。detail 通常需要更多 replay，开销更大；缺少 JIT line info 时不保证自动关联到 Python/Triton 源码行。

### 工作负载、输出和重放控制

| 参数 | 默认值 | 作用 |
|---|---|---|
| --case ID | PROFILE_CASE 或 w128_o128 | 冻结 case |
| --batch-size N | PROFILE_BATCH_SIZE 或 1 | 每次 generate 请求数 |
| --batch-tokens N | PROFILE_BATCH_TOKENS 或 2048 | 引擎 token budget |
| --warmup N | 5 | 预热 generate 次数，至少 5 |
| --requests-file PATH | REQUESTS 或项目 data/frozen/requests.jsonl | 必须保留对应冻结快照 |
| --output-dir PATH | NCU_OUT 或项目 log/ncu_offline | 唯一前缀的报告/CSV/日志等目录 |
| --cache-control all / none | NCU_CACHE_CONTROL 或 all | all 清理 replay 缓存；none 不主动清缓存，均不保证复现原 NSYS 缓存状态 |
| --clock-control none / base | NCU_CLOCK_CONTROL 或 none | none 不由 NCU 锁频；base 锁基准频率，对照时两组应一致 |
| -h / --help | — | 显示帮助，不启动工具或模型 |

NCU 每次只提交 1 次正式 generate。PROFILE_COUNT 仅适用于 NSYS，不改变 NCU 的正式请求数。NCU 固定使用 --target-processes all、--profile-from-start off、--replay-mode kernel、--graph-profiling node；这些由脚本保持，不与 basic/detail 的 section 粒度混淆。

## 4. 完整示例

### 用实际 kernel 名采指定调用

~~~bash
# 原 FP8 decode：跳过前三个同名矩阵调用，分析第四个。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --level detail \
  --kernel-name '_w8a8_triton_block_scaled_mm' \
  --launch-skip 3 --launch-count 1

# 同一 kernel 家族采 4 个匹配调用，先做轻量筛选。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model fp8-local --phase decode --level basic \
  --kernel-regex '_w8a8_triton_block_scaled_mm' \
  --launch-count 4

# BF16 prefill 的 CUTLASS 家族；模板参数等无需手写完整。
bash capstone/quantization_profiling/scripts/profile_offline_ncu.sh \
  --model bf16 --phase prefill --level detail \
  --kernel-regex 'cutlass::Kernel2<cutlass_80_tensorop_bf16_' \
  --launch-skip 2 --launch-count 1
~~~

### 同一工作负载的 NSYS 对照

~~~bash
bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh \
  --model fp8-local --mode graph --scope steady \
  --case w128_o128 --count 1 --batch-size 1 --batch-tokens 2048

bash capstone/quantization_profiling/scripts/profile_offline_nsys.sh \
  --model bf16 --mode eager --scope all \
  --case w128_o128 --count 1 --batch-size 1 --batch-tokens 2048
~~~

## 5. 环境变量与命令行优先级

| 环境变量 | 作用 | 对应命令行 |
|---|---|---|
| MODEL_ROOT | 四种模型的共同权重根目录，默认 /workspace/weights_pub | 通过环境设置 |
| REQUESTS | 冻结 requests.jsonl | 两者 --requests-file |
| PROFILE_CASE | 冻结 case | 两者 --case |
| PROFILE_BATCH_SIZE | 每次 generate 请求数 | 两者 --batch-size |
| PROFILE_BATCH_TOKENS | 引擎 token budget | 两者 --batch-tokens |
| PROFILE_COUNT | NSYS 正式 generate 次数 | NSYS --count |
| NSYS_OUT / NCU_OUT | 相应工具输出目录 | --output-dir |
| NCU_GRAPH | 1=graph，0=eager | NCU --mode |
| NCU_CACHE_CONTROL | replay cache 设置 | NCU --cache-control |
| NCU_CLOCK_CONTROL | 时钟设置 | NCU --clock-control |
| NSYS_BIN / NCU_BIN | 工具可执行文件路径 | 通过环境设置 |
| PYTHON_BIN | 原 vLLM 环境的 Python 路径 | 通过环境设置 |
| QUANT_PROFILE_LOG_EVERY_N_STEPS | worker decode 日志打印间隔，默认 32；NVTX 仍标记每步 | 通过环境设置 |

例如环境 PROFILE_BATCH_TOKENS=2048，但命令传 --batch-tokens 1024，则以 1024 为准。--mode graph 可以覆盖 NCU_GRAPH=0；环境变量仍应使用合法值。输出 settings.txt 会保存最终匹配表达式、name base、调用数量、窗口和 workload 参数，workload.json 保存引擎配置与快照身份。

## 6. 结果核对

先从 NSYS 确认当前离线入口的真实 kernel，再复制名字或构造正则。精确名匹配不到时，核对 name base、模板/参数是否完整、目标 phase、profiler window 和所选后端；不要把零命中解释为模型没有运行。

basic/detail 输出均保留 ncu-rep、metrics.csv、details.txt、workload.json、events 与日志。NCU replay 的时间和缓存/时钟条件与 NSYS 不同；该工具用于解释具体 kernel，不用进程总耗时替代离线推理延迟。

官方过滤语义依据：[NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。本页的默认值和参数组合以项目脚本为准。
