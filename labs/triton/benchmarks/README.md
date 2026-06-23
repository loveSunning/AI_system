# Benchmarks

这里保存 benchmark 脚本输出和可复现实验说明。建议 CSV 字段固定为：

```text
stage,op,impl,shape,dtype,config,warmup,iters,avg_ms,min_ms,max_ms,throughput,unit,reference,device,notes
```

推荐文件：

- `w09_vector_add_softmax.csv`
- `w10_matmul_autotune.csv`
- `w11_persistent_matmul.csv`
- `w12_fused_ops.csv`
- `w13_online_softmax.csv`
- `w14_attention_forward.csv`

同一个 CSV 中保留 PyTorch、cuBLAS 或手写 CUDA reference 的结果，避免只看 Triton 单点数字。
