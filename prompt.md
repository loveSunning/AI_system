工作任务：
    1. gemm中实现第三个版本。在tiled_gemm_register.cu的基础上 实现 gemm_dbuffer_vload.cu版本。如要考虑数据的dobule buffer加载，包括sharememory和register的dobule buffer。
    支持 blockm blockn 取 32  64  128  blockk  8 16 32 等。 vector load用  float4.  register tile 主要支持 4*4 8*8 ， 最好一个block中的线程数是256. 考虑各个维度的数据对齐。
    同时实现kernel核函数和调用函数。 把调用的cmd命令写到readme中  包括ncu的性能分析命令写到 profile/gemm/readme中
    



