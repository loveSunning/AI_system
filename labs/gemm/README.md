# GEMM 深入

对应 `W05-W08`。

建议把这里作为 `naive GEMM -> tiled GEMM -> register blocking -> Tensor Core demo -> autotune` 的主阵地。

推荐里程碑：

- `sgemm_v1.cu`：最小 tiled GEMM。
- `sgemm_v2.cu`：padding / vectorized load / bank conflict 处理。
- `wmma_demo.cu`：最小 Tensor Core matmul。
- `autotune/`：参数扫描脚本与结果表。
