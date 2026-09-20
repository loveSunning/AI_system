# P3 vLLM、SGLang 服务与训练架构闭环

来源：`00.AI-system-plan-optimized-v3.xlsx` / `案例3_Runtime与训练闭环`。以下为计划任务，状态沿用表格。

周计划与统一评测口径见 [W29–W52 计划](../../docs/learning-plan-w29-w52.md)。

| 任务ID | 周次 | 实施任务 | 具体步骤 | 交付物 | 验收标准 | 资源与依赖 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P3-01 | W31 | API与流式客户端 | 部署Qwen3-8B，固定采样/长度；记录请求到首token及完成时间 | server/client、metrics.jsonl | 可流式生成；EOS与长度截断记录明确 | 复用P1 BF16或W4A16 | 未开始 |
| P3-02 | W32 | 调度消融 | batch/KV预算/chunked prefill/prefix cache/Graph每次改一个 | 参数矩阵、原始CSV | 展示吞吐与p95 TTFT/TPOT变化 | 先并发1/8/16；32按显存扩展 | 未开始 |
| P3-03 | W35 | SGLang对照 | 同模型同精度同请求顺序；重复前缀与随机前缀；冷/热缓存分开 | 双runtime报告 | 质量、TTFT、TPOT、吞吐口径一致 | 不预设某框架更快 | 未开始 |
| P3-04 | W36 | 推理项目交付 | 核验实际attention backend；连接P2测量结果解释服务瓶颈 | Project3推理报告、演示 | 15分钟讲清API→调度→KV→kernel；开始持续投递 | FlashInfer是kernel库，不等于完整runtime | 未开始 |
| P3-05 | W39–W40 | SFT到Serving | 固定小数据集LoRA/QLoRA；保存/恢复adapter；Liger A/B；部署适配后模型 | 训练配置、checkpoint、服务回归 | 至少100 optimizer steps；留出质量回归和恢复测试 | 微调不替代预训练；无需8B全参 | 未开始 |
| P3-06 | W41 | DDP/NCCL | 小模型2卡；对齐有效batch；测通信与step时间 | collective/训练trace | 解释AllReduce/AllGather/ReduceScatter用途 | 无多卡则标待验证 | 未开始 |
| P3-07 | W43 | MCore单卡闭环 | 用官方组件搭小GPT；保存并恢复optimizer/checkpoint；记录格式 | MCore训练/恢复日志 | 100 step、loss有限、恢复状态正确 | 先小模型；不是Qwen3权重转换已完成 | 未开始 |
| P3-08 | W44 | MCore多卡扩展 | 同型号2卡TP或DP；固定global batch与序列；可选Qwen3导入 | 多卡报告、兼容性记录 | 真实多卡step/通信数据；说明Qwen3导入是否验证 | 2卡不能完成所有TP/PP/DP组合；EP仅概念 | 未开始 |
| P3-09 | W49 | 稳定性 | 30分钟混合请求压测；测试取消/超时/OOM后的恢复 | 运行手册、监控样例 | 错误率、重启恢复、内存增长有记录 | 不把本地实验声明为生产规模 | 未开始 |
