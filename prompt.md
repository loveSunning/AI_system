工作任务：
      1.在D:\workspace\learing\AI_system\labs\hgemm中实现hgemm的各种kernals。并且生成 hgemm_benchmark_lab.cpp对比各个核函数的性能。每个核函数的各种block_tile和regesiter的尺寸写死的，不用额外传参。输入输出矩阵的尺寸不固定。
      2.参考https://github.com/xlite-dev/HGEMM项目实现下面的调用发射函数和核函数：
            void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c); 
            void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m16n16k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m16n16k16_mma4x2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m16n16k16_mma4x2_warp2x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);                                                        
            void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
            void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
        但是输入输出的参数不用torch::Tensor格式，直接用half的格式。
    3.在D:\workspace\learing\AI_system\docs\profiling\hgemm实现使用文档。参考D:\workspace\learing\AI_system\docs\profiling\gemm\README.md，命令行输入的尺寸参数默认 m,n,k 4096 4096 4096.并且每个kernal函数都要生成ncu的性能分析命令。
