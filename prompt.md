工作任务：


    tiled GEMM v1（shared memory）实现tiled_gemm_v1_kernel  tiled_gemm_v1_cuda 是不是写在这个 D:\workspace\learing\AI_system\labs\gemm 里面最好。而不是代码实现都写在src/kernels里，labs/gemm只写了一个main函数。 原来的设计架构是这样吗，核心代码只写在src，labs里只写 main。正常逻辑是不是实验代码写在labs里，等这个实验成熟了，再集成到src中。现在这个逻辑比较换乱。基于最优实践看看项目的目录结构是不是要重构。如果需要，就重构，并给出理由和设计思想。

    如果按照现在的写法，labs/perf_engineering 和和labs/gemm是冗余的。
