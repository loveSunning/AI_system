add_library(ai_system_gemm_lab)
target_sources(ai_system_gemm_lab PRIVATE gemm_lab.cpp)

if(AI_SYSTEM_ENABLE_CUDA)
    target_sources(ai_system_gemm_lab PRIVATE gemm_lab.cu tiled_gemm_block.cu tiled_gemm_register.cu gemm_dbuffer_vload.cu)
    ai_system_configure_cuda_target(ai_system_gemm_lab)
endif()

target_include_directories(ai_system_gemm_lab PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ai_system_gemm_lab PUBLIC ai_system_core)
target_compile_features(ai_system_gemm_lab PUBLIC cxx_std_20)
ai_system_enable_warnings(ai_system_gemm_lab)

add_executable(sgemm_benchmark_lab sgemm_benchmark_lab.cpp)
target_link_libraries(sgemm_benchmark_lab PRIVATE ai_system_gemm_lab)
target_compile_features(sgemm_benchmark_lab PRIVATE cxx_std_20)
ai_system_enable_warnings(sgemm_benchmark_lab)
