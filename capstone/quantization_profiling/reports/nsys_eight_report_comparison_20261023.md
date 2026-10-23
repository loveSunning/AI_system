# 八份 Nsight Systems 报告的详细对比分析

分析日期：2026-10-23。输入目录：`capstone/quantization_profiling/log/profile`。

本次读取了八份 `.nsys-rep`，分别导出 SQLite，检查内嵌命令、请求指纹、GPU、CUDA API、kernel、Graph 元数据与采集告警；没有运行 benchmark、模型推理或量化。

## 1. 结论与可比较范围

**八份文件不是八份均具备 GPU 性能数据的报告。**

- 原四份 eager 报告有约 36 万至 44 万条 kernel，可分析已记录的推理区间，但存在 CUDA 事件可能不完整的警告。
- 新四份 no_eager 报告确认请求了 `enforce_eager=false`，stdout 记录预热 5/5、正式请求 1/1 成功；但 GPU kernel、GPU Graph 执行和内存传输活动均没有记录下来。
- 新报告各有 300 条 `CUDA_GRAPH_EVENTS`，具体是 225 条 Graph Creation 和 75 条 GraphExec Creation。它们都是零时长资源标记，不能当成 300 次回放，也不能计算 Graph 执行时间。
- 所以目前无法给出任何模型的 eager→Graph 加速比，也无法用新增四份报告判断 Graph 下哪个精度更快。
- eager 中两个 FP8 的线性层耗时几乎一致；相对 BF16，当前 Triton FP8 路径没有体现线性层加速，且存在更多无已记录 kernel 的时间。
- AWQ 的 Marlin 线性层确实更快，但这个优势在当前 profiler 运行的整体周期中被其他间隔掩盖。

本报告中的墙钟、kernel duration、CUDA API duration分别解释，不把 profiler 数据作为无 profiler 的正式吞吐结果。

## 2. 八份文件和采集覆盖

第一组文件后缀：`w128_o128_20261023T010402Z.nsys-rep`；第二组以 `no_eager_` 开头，后缀为 `w128_o128_20261023T031234Z.nsys-rep`。

下表“0/无表”表示文件没有记录对应数据，不表示程序没有执行计算。

| 报告标识 | 文件 MB | eager | 记录总长 s | kernel 条数 | CUDA API 条数 | memcpy 条数 | Graph 资源标记 | GPU Graph 执行记录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 19.310 | true | 114.655 | 358863 | 404971 | 33189 | 0 | 0 |
| awq | 18.598 | true | 66.201 | 358053 | 405460 | 34641 | 0 | 0 |
| fp8-official | 23.266 | true | 89.457 | 444491 | 487919 | 32189 | 0 | 0 |
| fp8-local | 23.311 | true | 89.164 | 444490 | 487921 | 32189 | 0 | 0 |
| no_eager_bf16 | 0.162 | false | 73.155 | 0 | 5 | 0 | 300 | 0 |
| no_eager_awq | 0.170 | false | 78.706 | 0 | 5 | 0 | 300 | 0 |
| no_eager_fp8-official | 0.182 | false | 108.696 | 0 | 5 | 0 | 300 | 0 |
| no_eager_fp8-local | 0.180 | false | 106.014 | 0 | 5 | 0 | 300 | 0 |


总记录长度包括加载、编译/初始化、预热、推理与服务退出。BF16 从约 114.7 s 变为 73.2 s，不意味着 Graph 推理快了 1.57 倍；FP8 总记录变长也不能证明 Graph 推理变慢。

### 2.1 配对条件核对

八份内嵌 stdout 均记录：

- 负载 `w128_o128`，1 轮，5 个预热请求，1 个正式请求。
- 请求子集 SHA256 为 `a9269fa978821ab7bfa6f53eeced1284453f38798fecf4ec784f3dcb6c479923`，selected IDs 一致。
- GPU 为同一 UUID 的 NVIDIA GeForce RTX 4090 D，compute capability 8.9，114 SM。
- 驱动记录为 580.82.09，CUDA driver API 版本为 13000；这不是 PyTorch wheel 自带 CUDA runtime 的版本。
- 每模型两次启动命令，去掉 `--enforce-eager` / `--no-enforce-eager` 后一致。模型路径、dtype、量化请求、max_model_len=256、max_num_seqs=1、max_num_batched_tokens=2048 一致。
- FP8 均请求后端 auto，`VLLM_TEST_FORCE_FP8_MARLIN=0`。

这些证据确认请求和启动参数可配对，但报告没有附带完整 environment.json、run.json、requests.jsonl、summary.json 和 server.log。相同权重路径不证明文件内容/revision 没变，相同 Python 路径也不证明所有包版本没变；没有据此扩展结论。

**八份采集元数据的 CUDA_GRAPH_TRACE_OPTIONS:MODE 都为 Graph。** eager=true 时该选项不会强制模型使用 Graph。当前 quickstart 中写着 node，也不能反推这八份历史报告就是 node 粒度。

## 3. 新四份 Graph 报告：实际记录了什么

### 3.1 300 条标记的真实含义

| 模型 | Graph Creation | GraphExec Creation | 资源标记出现区间 s | 标记 duration 合计 ns |
| --- | --- | --- | --- | --- |
| BF16 | 225 | 75 | 54.765971–54.922543 | 0 |
| AWQ | 225 | 75 | 68.208875–68.382342 | 0 |
| 官方 FP8 | 225 | 75 | 89.633863–89.903980 | 0 |
| 本地 FP8 | 225 | 75 | 86.182582–86.461564 | 0 |


这是 Graph 对象与可执行 Graph 对象的资源元数据。表中区间表示首末标记的时间跨度，不是总捕获耗时、实例化耗时或请求延迟；75 个 GraphExec 也不等于 75 次请求、75 个输出 token 或 75 次实际回放。

### 3.2 为什么这不是“正常 Graph 模式隐藏了单个 kernel”就能解释

正常 graph 粒度可能不记录图内每个 kernel，因此仅仅缺少 `CUPTI_ACTIVITY_KIND_KERNEL` 不能证明采集失败。但新增四份报告同时出现：

1. 没有 GPU kernel 表。
2. 没有 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 整图 GPU 活动表。
3. 没有 memcpy 活动表。
4. CUDA API 表每份只有 5 条，都是查询/初始化调用，没有记录 cudaGraphLaunch/cuGraphLaunch，也没有常规推理 launch 序列。
5. 实际产生 Graph 资源标记的 worker 有 `Not all CUDA events might have been collected.` 告警。

每份仅有的 5 条 API 都是：cuModuleGetLoadingMode 2 次、cudaGetDeviceProperties 2 次、cudaMemGetInfo 1 次。其统计中 cudaMemGetInfo 可能占 98%，但这只是“被记录到的 5 条查询”的占比，不能解释成推理主要在查询显存。

本机官方 `nsys stats --report cuda_gpu_kern_sum` 对四份新文件均返回 SKIPPED，提示不包含 CUDA kernel data；对应日志已经保存到派生目录。

NVIDIA 将整图执行记录与 Graph 对象元数据分开存储，见 [SQLite/后处理文档](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。CUPTI 对 graph 粒度的说明也明确区分整图 activity 和逐节点 activity，见 [CUPTI Usage](https://docs.nvidia.com/cupti/main/main.html)。

### 3.3 当前无法回答的 Graph 性能问题

- Graph 回放是否真正覆盖了 decode？未记录回放事件，无法核实。
- Graph 下 FP8 是否仍走 Triton？启动请求一致，但缺实际 kernel，无法由这四份确认。
- Graph 是否消除了 FP8 的量化 launch 开销？缺少回放和节点数据，无法核实。
- Graph 是否让 AWQ 的短 GEMM 优势转化为吞吐？缺少请求性能与 GPU 时序，无法计算。

不能将缺失数据填成 0 ms，也不能仅凭报告文件从 20 MB 缩小到 0.17 MB 判断优化成功。

## 4. eager 对照方法与实际时间窗

### 4.1 选取同等计算量

先按 prefill 线性层的特定 kernel/启动形状识别六簇请求，再利用每周期出现一次、GridX=18992 的 cuBLAS GEMV作为边界：从 prefill 后 GEMV 序列的第 11 次结束之后，到第 31 次结束，选取 20 个连续周期。该边界根据模式推定，不是原始 NVTX 阶段标签。

每个窗口核对：

- 36 层 × 每层 4 个线性调用 × 20 周期 = 2880 个 Transformer 线性层调用。
- FP8 额外有 2880 个 per_token_group_quant_8bit_kernel。
- 同形状输出 GEMV 20 次；结合模型结构推测为输出头，尚无算子参数证据确认。

下文称“正式窗口”，指最后一个正式请求中的局部 20 周期，不是完整请求。起点比前一边界结束晚 1 ns，避免 nsys 相交筛选重复计入前一个 GEMV。

| 模型 | 正式窗口 s | 已记录 kernel 数 | 平均周期 ms | 周期中位数 ms | 周期 min–max ms |
| --- | --- | --- | --- | --- | --- |
| BF16 | 111.674046447–112.122016952 | 9620 | 22.399 | 21.440 | 21.115–36.274 |
| AWQ | 63.130370244–63.612948946 | 9620 | 24.129 | 23.763 | 23.194–27.561 |
| 官方 FP8 | 84.619835327–85.363512007 | 12500 | 37.184 | 37.128 | 36.911–37.743 |
| 本地 FP8 | 84.513019558–85.266944467 | 12500 | 37.696 | 37.502 | 37.278–39.262 |


这些周期是 GPU 模式边界之间的墙钟，不是 HTTP 文本事件 TPOT。只采一轮，不能评价跨轮稳定性；预热请求也不能冒充五轮正式重复实验。

### 4.2 正式窗口的耗时分解

kernel 累计 duration 与 kernel 区间并集在这些窗口中相同，未发现已记录 kernel 之间重叠，因此下表可以按墙钟分解。若新数据有跨 stream 重叠，不能再直接累加代替并集。

| 模型 | 墙钟 ms | Transformer 线性 ms | 单独 GEMV ms | 激活量化 ms | 其它 kernel ms | 无已记录 kernel ms | 含传输的 CUDA 时间覆盖 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 447.971 | 297.863 | 25.975 | 0.000 | 16.838 | 107.296 | 76.218 |
| AWQ | 482.579 | 89.799 | 25.982 | 0.000 | 19.296 | 347.502 | 28.157 |
| 官方 FP8 | 743.677 | 322.462 | 25.973 | 4.738 | 18.027 | 372.476 | 50.015 |
| 本地 FP8 | 753.925 | 321.558 | 25.978 | 4.764 | 18.129 | 383.496 | 49.232 |


“无已记录 kernel”可能包含 CPU/框架执行、同步、调度、内存传输、未追踪活动或未刷新事件，不能直接称为 CPU 计算时间。CUDA 时间覆盖率也不是 SM 利用率、Tensor Core 利用率或 occupancy。

![eager 诊断对照](../../../out/quantization_profiling/profile_comparison_8_20261023/eager_comparison.png)

图中右侧 1–5 是预热请求，6 是正式请求；每个点都只取同样 20 个周期。Graph 数据不可用，没有画成零值。

## 5. 两个 FP8 的详细比较

### 5.1 相同路径，主要 GPU 成本几乎相同

两者均实际记录到 `_w8a8_triton_block_scaled_mm` 和 `per_token_group_quant_8bit_kernel`，eager 中没有 FP8 Marlin 路径的证据。

正式窗口中，本地 FP8 的线性层为 321.558 ms，官方为 322.462 ms，本地约低 0.28%；激活量化为 4.764 ms 对 4.738 ms，差别约 0.026 ms。这不支持“本地量化导出的模型有显著额外 GPU kernel 成本”的判断。

本地整体窗口反而比官方长约 10.248 ms，主要体现为无已记录 kernel 的时间多约 11.020 ms；线性层还略短约 0.904 ms。只凭本次单窗口，不能把这些间隔差异归因于权重量化质量或格式。

### 5.2 同名 GEMM 要按启动形状拆开

| Grid | Block | 各自次数 | 官方 Avg us | 本地 Avg us | 官方累计 ms | 本地累计 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 32×1×1 | 128×1×1 | 1440 | 99.093 | 98.771 | 142.694 | 142.230 |
| 48×1×1 | 128×1×1 | 720 | 62.576 | 62.552 | 45.055 | 45.037 |
| 192×1×1 | 128×1×1 | 720 | 187.102 | 186.515 | 134.714 | 134.291 |


Grid=32 还混合两个线性层形状；Grid/Block 不是 GEMM 的 M/N/K。三组时间都接近，说明两模型的相同启动形状表现一致。

4090 D 有 114 SM，Grid=32/48 的单次启动不足以同时覆盖全部 SM，是并发 1、小 M 下并行度受限的线索。FP8 Triton 还记录了 238 registers/thread、25088 B dynamic shared memory；是否 spill、实际 occupancy、DRAM 是否饱和，必须由 NCU验证，不能仅凭这些启动参数判定。

### 5.3 相对 BF16 的降速具体体现在哪里

正式 20 周期，本地 FP8 相对 BF16：

```text
窗口墙钟增加                 305.954 ms
Transformer 线性层增加        23.696 ms
额外激活量化 kernel            4.764 ms
无已记录 kernel 的时间增加   276.200 ms
剩余差额来自输出 GEMV/其它 kernel 的小幅变化
```

从已记录时间分解看，约 90% 的窗口差额落在“无已记录 kernel”部分。这不是已证明 90% 都是 Python 开销；事件丢失、追踪开销等仍可能影响该分解。

FP8 线性层相对 BF16 约慢 8%，所以当前路径也没有体现单独线性计算收益。可能与小 M 下的 kernel tiling、调度、访存/计算效率有关，具体硬件原因尚待 NCU。

### 5.4 量化成本要分 GPU duration 与逐次发射

激活量化约占 FP8 kernel 累计时间的 1.28%，占局部墙钟约 0.63%。其 GPU duration 小，但每周期额外 144 次调用，使总 kernel 数从 9620 增至 12500，约增加 29.94%。

不能只看这 4.7 ms就忽略量化路径的发射和调度影响，也不能把所有无 kernel 间隔都归因于这些量化调用。需要检查逐次 CPU launch 与 GPU 执行关联，或使用完整 Graph 数据进行受控对照。

## 6. BF16 与 AWQ 对照

### 6.1 AWQ：矩阵计算加速是真的，整体加速尚未体现

正式窗口，AWQ 的 2880 次 Marlin 线性层累计 89.799 ms，对照 BF16 297.863 ms；该部分约快 3.32 倍、耗时降低约 69.85%。

但整个窗口是 482.579 ms 对 447.971 ms，AWQ 反而多约 34.608 ms。无已记录 kernel 部分多了约 240.206 ms，抵消了线性层节省的约 208.064 ms。

这正是适合进一步检查 CPU 提交节奏、同步和 Graph 覆盖的情形。不能由此得出无 profiler 的 AWQ 吞吐一定比 BF16 低，也不能直接宣称 Graph 一定能消除全部间隔。

AWQ 的 awq_marlin_repack_kernel 是加载/转换阶段的成本，不属于每 token 的反量化成本；Marlin 自身执行的融合解包/计算则包含在其 kernel duration 内。

### 6.2 输出 GEMV 和 attention 的相对占比变化

每周期一次的同形状 BF16 GEMV，在四模型中都约 1.299 ms。20 次累计约 25.98 ms，几乎不随量化精度变化。

它约占 AWQ 已记录 kernel 时间的 19.2%，却只占 FP8 的约 7%；这不是 AWQ 的输出头突然变慢，而是 Marlin 主线缩短后，固定成本的相对占比提高。进一步加速主干后，应重新计算输出头、attention、normalization 的收益上限。

### 6.3 prefill 能说到什么程度

| 模型 | 正式请求 prefill 线性层簇首尾跨度 ms |
| --- | --- |
| BF16 | 20.496 |
| AWQ | 24.147 |
| 官方 FP8 | 33.575 |
| 本地 FP8 | 33.880 |


该跨度穿过 144 个线性层及它们之间的其它工作，不是 GPU GEMM duration 之和，也不是 TTFT 或完整 prefill 时间。这里只是 w128 短输入；不能外推此前 w4096/w8192 的 prefill 表现。Graph 四组没有对应活动，无法比较 Graph 对 prefill 的影响。

## 7. CPU 发射、队列和 GPU duration

正式窗口的热点平均值如下。A=CPU API duration，Q=API 返回到 GPU 开始，K=kernel duration；Q 只对非负记录取平均，不能当成 HTTP 队列时间。

| 模型 | kernel | 样本数 | Q 样本数 | A Avg us | Q Avg us | K Avg us |
| --- | --- | --- | --- | --- | --- | --- |
| AWQ | Marlin | 2880 | 2857 | 5.568 | 0.619 | 31.180 |
| 官方 FP8 | _w8a8_triton_block_scaled_mm | 2880 | 2776 | 6.317 | 6.236 | 111.966 |
| 官方 FP8 | per_token_group_quant_8bit_kernel | 2880 | 2802 | 6.255 | 46.516 | 1.645 |
| 本地 FP8 | _w8a8_triton_block_scaled_mm | 2880 | 2877 | 6.806 | 5.443 | 111.652 |
| 本地 FP8 | per_token_group_quant_8bit_kernel | 2880 | 2871 | 6.803 | 44.120 | 1.654 |


AWQ 的 Marlin Q 很小，结合较低的 CUDA 时间覆盖率，值得检查 GPU 是否在等待新提交。FP8 激活量化 Q 大，可能是在等待前面的 GEMM；不能认为把 Q 变成 0 就会加速。

量化 kernel 的 CPU API 平均约 6–7 us，而其 GPU duration 约 1.65 us，反映出小 kernel 逐次发射成本值得关注。但两者可以重叠，A+Q+K 不是请求延迟。

按“完全落在窗口内”的 CUDA launch API duration 累加，BF16约 60.1 ms、AWQ约 53.4 ms、官方 FP8约 78.6 ms、本地 FP8约 84.9 ms。这些 CPU duration不包含 API 调用之间的所有框架工作，也不能加到 GPU duration 后得到 E2E。

API 在边界外开始、而 kernel 在边界内执行时，API 计数可能少 1–2 条；这与官方 nsys 的相交筛选规则略有不同。kernel 数与累计 duration已另用官方 CSV逐一核对。

## 8. 六次请求的局部窗口，避免只看单个点

单位：每个 20 周期窗口的平均周期 ms。列 1–5 是预热，列 6 是正式请求。

| 模型 | 预热1 | 预热2 | 预热3 | 预热4 | 预热5 | 正式1 |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | 36.535 | 23.575 | 22.820 | 21.669 | 21.310 | 22.399 |
| AWQ | 23.592 | 23.265 | 23.283 | 23.678 | 23.237 | 24.129 |
| 官方 FP8 | 42.439 | 38.067 | 36.939 | 37.691 | 37.033 | 37.184 |
| 本地 FP8 | 39.887 | 38.485 | 39.725 | 42.698 | 37.807 | 37.696 |


BF16 第一次预热较慢，但线性层累计时间仍约 297.8 ms，与后续接近，变化主要落在间隔。FP8 也有局部较慢的预热窗口，不能据此证明本地模型持续比官方模型慢。

这些是同一轮的不同请求，既包含启动后的热身过程，也可能含调度和 profiler 干扰；不是多轮独立重复，不能用于显著性检验或长期稳定性结论。

## 9. 采集缺失：证据、假设与验证顺序

### 9.1 已确认的证据

- 全八份都有实际 worker 的 CUDA 事件可能不完整警告。
- 全八份都有 CUDA driver 13.0 使用 12.9 追踪库的兼容提示，载入路径为 `/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/libcupti.so.12.9`。
- 新四份缺失的不仅是图内 kernel，还有整图执行和推理 CUDA API。
- 八份均记录 CUDA_FLUSH_PERIOD=0；CUDA_FLUSH_ON_CUDA_PROFILER_STOP=false。后者不意味着本次调用过 cudaProfilerStop，只是配置值。
- 仓库当前 stop_process() 等待 API 父进程后，尝试 SIGKILL 剩余同组 worker，未保证 worker 完成 profiler 刷新。

### 9.2 合理假设，但还不是根因结论

一种能解释现象的机制是：eager 产生大量活动，部分缓冲已满而先行写出，所以留下大量记录但尾部不完整；Graph 模式减少逐次活动，更多内容留在缓冲中，worker 异常结束时没有刷新，只留下通过不同通道记录的 Graph 资源标记。

这是依据数据形态与退出逻辑的假设。还可能有采集器/驱动在 Graph 路径的兼容问题。驱动警告在 eager/Graph两组都有，因此不能单靠这条共同警告断言它是两组差异的唯一原因。

### 9.3 建议的验证顺序（本次未执行重采）

1. 固定一个模型与现有 Graph 配置，使用完整支持目标驱动的采集器，并核对与其配套的 QdstrmImporter；升级查看器无法补回旧报告中的活动。
2. 让 profiler 在 GPU worker 仍存活时完成 stop/flush，再关闭服务；或者保证 worker 正常退出，避免父进程退出后立即强杀。相关调用必须作用于实际 GPU worker，而不是仅在 benchmark 主进程调用同步或 sleep。
3. 可独立增加 --cuda-flush-interval=1000 做数据保留对照。它会增加追踪开销，也不能保证保存最后一次刷新之后的数据。新旧结果不能直接作为同开销性能对照。
4. 先用 graph 粒度验证回放 activity 能记录，再用 node 粒度采集少量区间看实际热点。graph 与 node 的可见内容和追踪开销不同，应分别保存配置。
5. 重采成功后先验收数据覆盖，再比较性能。benchmark请求成功和profiler数据完整是两件事。

对 graph 粒度，至少应看到实际 GPU Graph activity和相应 launch序列；对 node粒度，应有实际节点 kernel及可关联的 Graph信息。不要因单个不使用CUDA的辅助进程警告而否定整个文件，重点检查真正的worker与请求范围。

## 10. 当前文件如何复核：命令与 GUI

### 10.1 Windows PowerShell：官方表

```powershell
$nsysTool = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe'
$analysisDir = 'D:\workspace\learing\AI_system\out\quantization_profiling\profile_comparison_8_20261023'

# 本地 FP8 的 eager 正式窗口
& $nsysTool stats --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,cuda_api_sum --timeunit us --filter-time=84.513019558s/85.266944467s "$analysisDir/fp8-local.sqlite"

# Graph 文件：会显示缺少 CUDA kernel data，只能得到少量查询 API
& $nsysTool stats --report cuda_gpu_kern_sum,cuda_api_sum --timeunit us "$analysisDir/no_eager_fp8-local.sqlite"
```

其它 eager窗口见第4节，换文件时同时换时间范围。数值采用会话时间；修改 GUI 时间原点后先对齐同一事件。

### 10.2 SQLite：证明 Graph 记录是什么

在任意 no_eager SQLite 中执行：

```sql
SELECT name FROM sqlite_master
WHERE type='table' AND (
  name LIKE 'CUPTI_ACTIVITY_KIND_%' OR name LIKE 'CUDA_GRAPH%'
);

SELECT s.value AS event_name,COUNT(*) AS records,
       SUM(g.end-g.start) AS recorded_duration_ns,
       MIN(g.start)/1e9 AS first_s,MAX(g.end)/1e9 AS last_s
FROM CUDA_GRAPH_EVENTS g
JOIN StringIds s ON s.id=g.nameId
GROUP BY g.nameId;

SELECT s.value AS api_name,COUNT(*) AS calls
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id=r.nameId
GROUP BY r.nameId;
```

第一条检查的是表是否存在，不能对一个不存在的 GPU Graph 表直接 COUNT后把 SQL错误当成“执行时间0”。

### 10.3 GUI 定位

- eager 本地 FP8：PID 440212，84.513019558–85.266944467 s；官方 FP8：PID 427569，84.619835327–85.363512007 s。
- eager BF16：PID 424274，111.674046447–112.122016952 s；AWQ：PID 437548，63.130370244–63.612948946 s。
- no_eager 的 Graph资源标记所在worker：BF16 505339、AWQ 511973、官方FP8 508612、本地FP8 515118。位置见第3节。
- 新报告先看 Diagnostics Summary，再在 Timeline/Events View 查看 Graph Creation 标记；看到这些标记不代表 GPU 整图耗时已经可见。
- eager 用 Show in Events View 查看具体 kernel，并关联 CPU launch；不要把缩放视野自动当成聚合表范围过滤。

## 11. 证据文件与复现

所有派生结果：`out/quantization_profiling/profile_comparison_8_20261023/`。

- manifest.json：输入文件名、大小和分析时刻 SHA256。
- inventory.csv：八份报告的数据覆盖与采集信息。
- graph_creation.csv：新四份Graph资源事件分类与范围。
- eager_decode_windows.csv：24个窗口的完整指标。
- eager_formal_kernel_groups.csv：四模型正式窗口的kernel分组。
- fp8_formal_gemm_grids.csv：两个FP8按Grid拆分的GEMM。
- analysis.json：配置、告警、API、窗口与聚合结果。
- *_formal20_*.csv：由官方nsys stats导出的四种统计表。
- no_eager_*_stats.log：四份新报告缺失kernel数据的官方工具输出。
- eager_comparison.png/.svg：可分享的静态对照图。

从仓库根目录复核：

```powershell
python out/quantization_profiling/profile_comparison_8_20261023/analyze_eight.py
python out/quantization_profiling/profile_comparison_8_20261023/plot_comparison.py
```

这两个脚本读取已导出的SQLite，不运行模型；第二个会交叉核对官方CSV中的kernel数和累计时间，再更新图。

后续最有价值的新增证据是四模型完整的Graph时序和配套请求记录。得到它们之前，能确定的是eager的实际热点与Graph采集缺口，不能给出Graph加速结论。
