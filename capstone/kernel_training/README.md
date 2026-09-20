# P2 Kernel、FlashInfer 与训练算子优化

来源：`00.AI-system-plan-optimized-v3.xlsx` / `案例2_Kernel与训练优化`。以下为计划任务，状态沿用表格。

周计划与统一评测口径见 [W29–W52 计划](../../docs/learning-plan-w29-w52.md)。

复用 `labs/triton`、`labs/cute`、`labs/cutlass` 和 `labs/flash_attention` 的既有成果；custom op 接入代码放在 `integrations/pytorch`。本目录汇总案例实验、配置与报告。

| 任务ID | 周次 | 实施任务 | 具体步骤 | 交付物 | 验收标准 | 资源与依赖 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2-01 | W33 | 继承FA2成果 | 整理已有FA2 forward/backward、CuTe layout、copy/MMA笔记，复用真实shape | FA2数据流与shape表 | 解释delta/dQ/dK/dV；不重复原28周内容 | 依赖P1模型配置 | 未开始 |
| P2-02 | W33 | FlashInfer Prefill | 对齐SDPA/FA2的mask、scale、dtype、GQA；分别测plan/run | prefill_bench.csv | 多shape输出误差通过，阈值和误差均记录 | 4090支持的后端 | 未开始 |
| P2-03 | W34 | Paged Decode | 构造变长page table；测试page_size、batch、长度与Graph开关 | decode_bench.csv、页表图 | 3种长度×2种batch；页边界正确 | 不含JIT/plan时间的热测和冷启动分报 | 未开始 |
| P2-04 | W37 | 两个custom op | 复用RMSNorm与RoPE或SwiGLU；封装dispatcher/fake/autograd；检查compile | 可安装包、单测 | 至少2个算子测评、1个模型集成；回退路径可用 | 浮点容差按dtype/参考实现预设 | 未开始 |
| P2-05 | W38 | Liger前反向对照 | 阅读融合交叉熵/RMSNorm；原生与Liger开关A/B，固定有效batch | loss/梯度日志、step/显存CSV | 梯度/loss不超约定容差；解释内存收益来源 | 8B单卡QLoRA短序列；不足先小block | 未开始 |
| P2-06 | W40 | 端到端收益 | 接入P3训练或推理，分别报告kernel与模型性能 | Project2_Report、补丁 | 功能回归通过；收益小/负收益也给出原因 | 不要求超过官方kernel | 未开始 |
| P2-07 | W42 | TE训练扩展 | 独立小Transformer block；BF16/FP8 recipe对照，保存amax/误差信息 | TE实验记录 | 运行100 step且loss有限；恢复/数值对照 | 先验证Ada与版本兼容；失败保留复现，不强制宣称FP8加速 | 未开始 |
