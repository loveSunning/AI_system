# W29–W52 学习计划

来源：`00.AI-system-plan-optimized-v3.xlsx`，以“学习排期表”及配套案例、验收工作表为准。W29 从 2026-09-21 开始，W52 于 2027-03-07 结束；周编号表示累计学习周，不是自然周。W01–W28 保留项目原有安排与记录。

每周约 20 小时：实现/实验 12h、源码与文档 4h、报告/投递 4h，共 24 周、480h；案例任务包含在周预算内。优先级 P0 表示就业必做、P1 表示训练/平台进阶、P2 表示延期选修；案例编号 P1–P4 是另一套编号，分别对应下表四个项目。

| 案例 | 项目目录 |
| --- | --- |
| P1 Qwen3-8B 量化与性能诊断 | [capstone/quantization_profiling](../capstone/quantization_profiling/README.md) |
| P2 Kernel、FlashInfer 与训练算子优化 | [capstone/kernel_training](../capstone/kernel_training/README.md) |
| P3 vLLM、SGLang 服务与训练架构闭环 | [capstone/runtime_training](../capstone/runtime_training/README.md) |
| P4 Qwen3-8B 国产平台迁移 | [capstone/domestic_migration](../capstone/domestic_migration/README.md) |

## 每周安排

| 模块 | 月份主题 | 周次 | 日期范围 | 学习重点/任务 | 阅读资料(URL) | 动手(Kernel/Project) | 验收/里程碑 | 风险/注意 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 模型基线 | Qwen3-8B 与量化 | W29 | 2026-09-21~2026-09-27 | 固定模型 revision、tokenizer、chat template、thinking 开关与生成参数；盘点 CUDA/driver/runtime | https://huggingface.co/Qwen/Qwen3-8B | P1：BF16 小并发基线和显存预算，记录环境与原始输出 | 5 组固定 workload 可复现；环境可重建 | 4090 单卡优先；长上下文/并发按显存实测 | P0；P1；未开始；20h |
| 性能诊断 | 量化与 Profiling | W30 | 2026-09-28~2026-10-04 | 先 nsys 分离 prefill/decode，再 ncu 抽样热点；运行一条 W4A16 路径 | https://docs.vllm.ai/en/latest/features/quantization/ | P1：BF16 对 W4A16；固定留出集质量回归 | 原始 CSV、trace、质量分数齐全；说明变快/变慢原因 | 校准集与评测集分离；INT8/FP8 后补 | P0；P1；未开始；20h |
| 推理服务 | vLLM 架构 | W31 | 2026-10-05~2026-10-11 | API server、engine、scheduler、KV manager、model runner、attention backend 调用链 | https://docs.vllm.ai/en/latest/ | P3：Qwen3-8B API、流式客户端、压测脚本 | 流式输出正常；记录 TTFT/TPOT/p95/错误率 | 启动参数按锁定版本；BF16 OOM 则用 P1 W4A16 | P0；P3；未开始；20h |
| 推理服务 | vLLM 调度与显存 | W32 | 2026-10-12~2026-10-18 | continuous batching、paged KV、chunked prefill、prefix caching、CUDA Graph | https://docs.vllm.ai/en/latest/ | P3：一次只改一个参数，比较短请求与长请求混合负载 | 给出吞吐和尾延迟取舍；建立可回滚配置 | W32 首轮投递：P1 报告 + P3 服务 demo | P0；P1/P3；未开始；20h |
| 推理算子 | FlashInfer Prefill | W33 | 2026-10-19~2026-10-25 | ragged/paged KV 格式、GQA、plan/run、workspace 与后端选择 | https://docs.flashinfer.ai/api/attention.html | P2：真实 Qwen3 shape，FlashInfer 对 SDPA/FA2 | 输出误差通过；plan 与 run 分开计时 | 读取模型配置，不能硬编码其他模型 head_dim | P0；P2；未开始；20h |
| 推理算子 | FlashInfer Decode | W34 | 2026-10-26~2026-11-01 | page indptr/indices/last_page_len；decode 与 batch decode；CUDA Graph 约束 | https://docs.flashinfer.ai/api/attention.html | P2：变长序列 paged attention，测 page size/批量变化 | 至少 3 种长度与 2 种 batch，记录误差、延迟、显存 | 不默认所有后端适配 4090；记录实际选中后端 | P0；P2；未开始；20h |
| 推理服务 | SGLang 系统对照 | W35 | 2026-11-02~2026-11-08 | RadixAttention、prefix cache、scheduler、backend；对照 vLLM 相同 workload | https://docs.sglang.ai/ | P3：重复前缀和无共享前缀两组；缓存冷/热分开 | 同模型/精度/请求序列对照；解释缓存命中收益 | 独立环境；不要用不同默认设置得出优劣 | P0；P3；未开始；20h |
| 就业交付 | 推理作品集 v1 | W36 | 2026-11-09~2026-11-15 | 整理 P1 与 P3 报告、环境锁定、架构图、失败配置和诊断方法 | https://docs.vllm.ai/en/latest/ | 录制 5 分钟服务演示；准备 15 分钟面试讲述 | P1/P3 核心验收后持续投递；整理反馈 | 本周用于补缺，不再加入新框架 | P0；P1/P3；未开始；20h |
| 框架接入 | PyTorch custom op | W37 | 2026-11-16~2026-11-22 | dispatcher、fake/meta、autograd、opcheck、torch.compile 边界 | https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html | P2：将已有 RMSNorm/RoPE 两个 kernel 封装，至少一个接入模型 | 正确性、异常 shape、eager/compile 通过；可安装 | 以已有代码为基础，不重写全部 FA2 | P0；P2；未开始；20h |
| 训练算子 | Liger Kernel | W38 | 2026-11-23~2026-11-29 | RMSNorm/SwiGLU/FusedLinearCrossEntropy 前反向与显存收益 | https://github.com/linkedin/Liger-Kernel | P2：同样训练配置对照原生和 Liger；用短序列 LoRA/QLoRA | loss/梯度一致性、step time、峰值显存有实测 | 分开测单算子和训练 step；先验证 Qwen3 patch | P0；P2；未开始；20h |
| 训练闭环 | Qwen3 微调与服务 | W39 | 2026-11-30~2026-12-06 | 数据清洗、label mask、LoRA/QLoRA、梯度累积、checkpoint 与评测 | https://github.com/linkedin/Liger-Kernel | P3：小数据 SFT，导出 adapter，加载回推理服务 | 训练、保存、恢复、服务、质量回归全链路跑通 | 单卡不要求 8B 全参训练；合并量化权重需单独验证 | P0；P3；未开始；20h |
| 训练闭环 | 训练性能优化 | W40 | 2026-12-07~2026-12-13 | 比较 Liger 开关、activation checkpoint、seq length；检查数据加载瓶颈 | https://github.com/linkedin/Liger-Kernel | P2/P3：固定有效 batch 和 token 口径，回归 serving | 训练报告含吞吐/显存/loss；持续投递推理/训练优化岗 | 只把实测结果写简历，不填预设加速倍数 | P0；P2/P3；未开始；20h |
| 训练架构 | torch.distributed 与 NCCL | W41 | 2026-12-14~2026-12-20 | rank/world size/process group、DDP、AllReduce/AllGather/ReduceScatter | https://docs.pytorch.org/docs/stable/distributed.html | P3 训练扩展：小模型 2 卡 DDP；单卡先验证初始化与 checkpoint | 可解释通信时间线、梯度同步及 global batch | 多卡性能用同型号设备；4090+5060 不作标准扩展性结论 | P1；P3；未开始；20h |
| 训练架构 | Transformer Engine | W42 | 2026-12-21~2026-12-27 | BF16 基线、FP8 scaling/amax/recipe、数值范围、fused module | https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/ | P2 训练扩展：小 Transformer block 的 BF16/FP8 前反向对照 | 记录硬件、TE recipe、误差、loss 与性能；失败有最小复现 | Ada 可支持 FP8，具体算子/版本先 smoke test；禁止默认套 Hopper 专用路径 | P1；P2；未开始；20h |
| 训练架构 | Megatron-Core 单卡 | W43 | 2026-12-28~2027-01-03 | model spec、GPTModel、optimizer、训练循环、分布式 checkpoint | https://docs.nvidia.com/megatron-core/developer-guide/latest/ | P3 训练扩展：先小 GPT 配置训练 100 step、保存并续训 | loss 有限；恢复后 step/optimizer 状态正确 | MCore 是组件库；Qwen3 8B 权重转换单列验证 | P1；P3；未开始；20h |
| 训练架构 | Megatron-Core 并行 | W44 | 2027-01-04~2027-01-10 | TP/PP/DP、sequence parallel、recompute、通信计算重叠 | https://docs.nvidia.com/megatron-core/developer-guide/latest/ | P3 训练扩展：同型号 2 卡小模型 TP 或 DP 对照 | 固定 global batch/序列/token 数，交付 trace 与扩展效率 | 无多卡则标待硬件验证，不能把单卡模拟写成多卡实测 | P1；P3；未开始；20h |
| 异构平台 | 国产芯片资源确认 | W45 | 2027-01-11~2027-01-17 | 确认一个能运行 Qwen3-8B 的国产平台；优先实际可用的 Ascend | https://docs.vllm.ai/projects/ascend/en/latest/ | P4：锁定卡型、驱动、SDK、框架/模型支持矩阵 | 拿到真实设备并完成官方 smoke test | 无硬件则做适配准备，P4 不标完成；不阻塞求职 | P1；P4；未开始；20h |
| 异构平台 | 模型部署与正确性 | W46 | 2027-01-18~2027-01-24 | 用平台官方适配链路部署 Qwen3-8B；明确 unsupported op 和 dtype | https://docs.vllm.ai/projects/ascend/en/latest/ | P4：复用 P1 数据与 P3 客户端；核验 tokenizer/输出 | 完成真实设备推理；固定质量集回归 | 优先相同精度；不支持量化的情况单独列组 | P1；P4；未开始；20h |
| 异构平台 | 跨平台 Profiling | W47 | 2027-01-25~2027-01-31 | 框架日志、设备 profiler、算子 fallback、搬运、KV/调度瓶颈 | https://docs.vllm.ai/projects/ascend/en/latest/ | P4：统一 workload，对照 NVIDIA 与国产卡 | 报告 TTFT/TPOT/吞吐/显存/质量与迁移工时 | 不同卡只比较部署方案，不冒充纯架构性能对照 | P1；P4；未开始；20h |
| 异构平台 | 优化与交付 | W48 | 2027-02-01~2027-02-07 | 选一项可测优化：batch/KV 参数或 RMSNorm 简单算子；保存回滚配置 | https://docs.vllm.ai/projects/ascend/en/latest/ | P4：适配报告、启动脚本、错误排查和 5 分钟演示 | 至少一个国产平台真实通过；第二平台选做 | 不再并行展开 TVM/MLIR 或多个国产生态 | P1；P4；未开始；20h |
| 系统整合 | 架构面试与可靠性 | W49 | 2027-02-08~2027-02-14 | 模型加载、调度、算子、通信、监控；超时/取消/限流与长压测 | https://docs.vllm.ai/en/latest/ | P3：30 分钟混合请求压测；整理 OOM/重启恢复案例 | 无未解释错误；指标可追溯；服务能恢复 | 高阶 PD 分离/投机解码只选一个概念演示 | P0；P3；未开始；20h |
| 作品集 | 四案例复现 | W50 | 2027-02-15~2027-02-21 | 统一入口、依赖锁定、模型 revision、数据许可、结果索引与报告 | https://huggingface.co/Qwen/Qwen3-8B | P1–P4：另一环境按 README 复跑代表 case | 每项目均有代码、数据、trace/日志、结论、局限 | 待硬件验证项明确标注；不得伪造完成 | P0；P1/P2/P3/P4；未开始；20h |
| 求职冲刺 | 按岗位补短板 | W51 | 2027-02-22~2027-02-28 | 推理岗讲 scheduler/KV；训练岗讲 Liger/TE/MCore；平台岗讲适配 | https://docs.pytorch.org/docs/stable/distributed.html | 模拟 3 场面试，按反馈修一个真实缺口 | 简历每项性能数字可定位到原始日志 | TVM/MLIR 只做分工认知，详细学习移到入职后 | P0；全部；未开始；20h |
| 缓冲收尾 | 交付与下一阶段 | W52 | 2027-03-01~2027-03-07 | 补失败实验、录演示、整理 PR/issue；制定入职后编译器学习清单 | https://docs.nvidia.com/megatron-core/developer-guide/latest/ | 四案例最终版；持续投递并记录反馈 | 核心案例可讲可复现；资源不足部分独立延期 | P2：TVM/TensorIR、MLIR Toy/pass、TPU-MLIR、RK3588 | P0；全部；未开始；20h |

## 验收清单

| 阶段 | 周次 | 必须产出 | 必须会解释 | 必须保存的证据 | 通过标准 |
| --- | --- | --- | --- | --- | --- |
| 量化与基线 | W29–W30 | P1 baseline/量化/质量报告 | prefill与decode瓶颈；权重与KV显存 | 环境、CSV、nsys/ncu、质量集hash | BF16与W4A16同口径；数据可追溯 |
| 服务与推理算子 | W31–W36 | P2 FlashInfer；P3 vLLM/SGLang | paged KV、调度、缓存、后端选择 | 启动脚本、冷/热缓存压测、误差日志 | TTFT/TPOT/p95/错误率完整；能解释取舍 |
| 算子与训练闭环 | W37–W40 | P2 custom op/Liger；P3 SFT | 梯度、融合、checkpoint、adapter服务 | 单测、梯度/loss、step time、显存 | 至少2个算子测评、1个接入；训练到服务跑通 |
| 训练架构 | W41–W44 | TE与MCore最小训练；多卡实验 | FP8 recipe、TP/PP/DP、NCCL、重算 | checkpoint恢复日志、trace、配置 | 单卡闭环通过；无多卡不得标多卡完成 |
| 国产迁移 | W45–W48 | P4真实设备部署与报告 | 框架/SDK/量化/通信适配差异 | 设备信息、推理日志、性能质量CSV | 至少一个国产平台成功；准备工作不等于部署完成 |
| 求职交付 | W49–W52 | 四项目README/复现脚本/演示 | 系统架构、瓶颈、优化前后与局限 | 每条简历数字对应实验ID | 另一环境复跑代表case；模拟面试能回答追问 |

## 框架学习与取舍

| 框架 | 优先级 | 安排 | 学习边界 | 实战关联 | 停止条件与资源限制 | 官方资料 |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM | P0 | W31–W32 主修，W49可靠性 | serving engine、scheduler、KV manager、model runner | P3：API/压测/参数消融 | 顺着一次请求读调用链；定位一处性能瓶颈 | https://docs.vllm.ai/en/latest/ |
| FlashInfer | P0 | W33–W34 | prefill/decode、paged KV、plan/run、CUDA Graph | P2：paged attention 正确性和性能矩阵 | 独立调用库，再核验runtime实际后端；不是服务框架 | https://docs.flashinfer.ai/api/attention.html |
| SGLang | P0 | W35–W36 | RadixAttention、缓存复用、调度与后端 | P3：与vLLM相同负载对照 | 至少一次缓存开关/冷热消融；不全量读源码 | https://docs.sglang.ai/ |
| Liger Kernel | P0 | W38–W40 | RMSNorm/SwiGLU/融合交叉熵前反向 | P2：训练算子；P3：SFT训练优化 | 理解融合节省什么中间张量；梯度与loss回归 | https://github.com/linkedin/Liger-Kernel |
| Transformer Engine | P1 | W42 | FP8 scaling、amax、recipe、fused layer | P2：小block BF16/FP8；P3训练扩展 | Ada/Hopper/Blackwell支持仍需版本和算子验证；与Liger分开A/B | https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/ |
| Megatron-Core | P1 | W43–W44 | GPT组件、TP/PP/DP、recompute、checkpoint | P3：小模型训练/恢复/多卡实验 | 先原生小GPT，再验证Qwen3导入；不要求单卡8B全参训练 | https://docs.nvidia.com/megatron-core/developer-guide/latest/ |
| PyTorch distributed/NCCL | P1 | W41，前期按需 | DDP、collectives、process group、通信profile | P3：实际2卡基准与通信时间线 | 同型号多卡做性能结论；异构卡只作功能实验 | https://docs.pytorch.org/docs/stable/distributed.html |
| CUDA/Triton/CuTe/FA2 | 已有基础 | W37定点复用 | 热点优化、custom op、编译接入 | P2：已有代码集成 | 保持算子优势，避免重复从头学FA2 | https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html |
| TVM / MLIR / TPU-MLIR | P2 延后 | W52后，按岗位选择 | IR、schedule、pass、lowering分工 | 一个最小实验即可起步 | 不设就业前硬门槛；不展开完整编译器路线 | https://tvm.apache.org/docs/<br>https://mlir.llvm.org/docs/Tutorials/Toy/ |
| 国产平台 | P1 | W45–W48 | 驱动/SDK/runtime/profiler/量化 | P4：一个可获得平台 | 平台具体支持在锁版本时再次确认 | https://docs.vllm.ai/projects/ascend/en/latest/ |

## 评测口径与资源

| 类别 | 执行口径 | 记录内容 | 验收或资源边界 |
| --- | --- | --- | --- |
| 版本 | 每个实验固定模型revision与依赖版本 | GPU/显存/driver/CUDA/runtime/backend/精度/量化配置 | 不同版本/后端的结果不能无标记混合 |
| 负载 | 输入/输出token：128/128、1024/128、4096/256、8192/128、1024/512；并发1/8/16 | 先最小配置，容量允许再测并发32；记录OOM和被拒请求 | 按实际token数统计；不把设置max_tokens当实际输出数 |
| 推理时延 | TTFT=提交到首token；单请求TPOT=(末token时间−首token时间)/(输出token数−1) | 输出≥2时计算TPOT；汇报每请求分布的p50/p95 | 单token TPOT留空；服务排队包含在TTFT中 |
| 吞吐 | 聚合输出token/s=所有成功请求输出token总和/测量墙钟时间 | 同时记录成功请求数、错误数、输入token与实际输出token | 不要把单请求速度直接当服务吞吐 |
| 重复与缓存 | 预热后至少3轮；固定请求序列；缓存冷/热分开 | 中位数、p95、离散程度；JIT/加载单独记录 | 性能波动大先排查温度、功耗、并发和后台任务 |
| 质量 | 固定100条留出任务，校准集分离；统一判分规则 | 准确率及差值（百分点）；可加固定语料PPL | 质量下降≤2个百分点仅建议目标，实验前确认；不保证实现 |
| 训练 | 固定序列、有效global batch、累积步数与非padding token口径 | step time、token/s、峰值allocated/reserved、loss、梯度和恢复状态 | Liger/TE分开A/B；不同时修改多个因素 |
| 计时 | 正式性能测量不开profiler；kernel计时做GPU同步 | nsys/ncu trace仅用于解释，另保存干净计时CSV | 单kernel提速与端到端收益分列 |
| 单卡 | 4090主要承担推理/算子和LoRA/QLoRA；5060按实际显存安排 | BF16权重只是显存组成之一；KV/workspace/激活需另留余量 | 不承诺8B全参训练；小block学习TE/MCore |
| 多卡 | 相同型号2卡起步，先通信再小模型训练 | GPU互联/PCIe拓扑、卡数、通信耗时、batch与有效token | 4090+5060可功能实验，不作标准扩展效率结论 |
| 国产平台 | 先确认一个可用且支持模型的平台 | 设备/SDK/框架版本、量化支持、实际生成和profile | 没有设备只完成准备，不编造部署或benchmark结果 |
| 来源与日期 | 参考用户两份工作簿；框架资料核验于2026-09-20 | 官方资料集中在“框架学习与取舍”和原学习排期URL列 | 版本与平台兼容在开始对应实验时再次核对 |

## 求职里程碑

| 时间 | 必备证据 | 目标岗位 | 面试主题 | 投递动作 |
| --- | --- | --- | --- | --- |
| W32（第4周） | P1量化/Profiling报告；P3 API demo与压测 | 推理部署/性能优化；GPU算子工程 | memory-bound、KV显存、真实热点与量化取舍 | 达到验收就试投递，记录岗位缺口 |
| W36（第8周） | FlashInfer实验；vLLM/SGLang同口径对照 | LLM推理优化 / AI Infra | 调度、paged KV、缓存、吞吐与尾延迟 | 持续投递；准备15分钟项目讲述 |
| W40（第12周） | custom op、Liger前反向与SFT到Serving | 训练性能优化 / 推理训练平台 | 融合、梯度、checkpoint、显存与质量 | 按真实结果增加训练项目经历 |
| W44（第16周） | TE/MCore小模型；多卡结果或明确待验证 | 训练框架工程 / 分布式训练 | TP/PP/DP、NCCL、FP8与重算 | 只描述实际完成深度，不自称大规模训练经验 |
| W48（第20周） | 一个国产平台真实部署、质量与性能报告 | 国产算力适配 / 异构计算 | runtime/SDK/profiler、fallback、迁移成本 | 突出原图像/端侧工程经验与LLM迁移能力 |
| W49–W52 | 四案例复现入口、原始日志、演示、技术取舍 | 按面试反馈聚焦一个方向 | 代码讲解、瓶颈追问、失败实验与回滚 | 每周固定投递/复盘；编译器学习按岗位再启动 |

## 延后学习

TVM/TensorIR、MLIR Toy/pass、TPU-MLIR 和 RK3588 移至 W52 后，按岗位需要启动。TVM/MLIR/TPU-MLIR 每周最多 2h，不作为投递前置。无多卡或国产设备时，对应实测任务独立延期并标注待硬件验证。
