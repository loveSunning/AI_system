add_library(ai_system_gemm_lab)
target_sources(ai_system_gemm_lab PRIVATE gemm_lab.cpp)

if(AI_SYSTEM_ENABLE_CUDA)
    target_sources(ai_system_gemm_lab PRIVATE gemm_lab.cu tiled_gemm_v1.cu)
    ai_system_configure_cuda_target(ai_system_gemm_lab)
endif()

target_include_directories(ai_system_gemm_lab PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ai_system_gemm_lab PUBLIC ai_system_core)
target_compile_features(ai_system_gemm_lab PUBLIC cxx_std_20)
ai_system_enable_warnings(ai_system_gemm_lab)

add_executable(gemm_tiled_v1_lab tiled_gemm_v1_lab.cpp)
target_link_libraries(gemm_tiled_v1_lab PRIVATE ai_system_gemm_lab)
target_compile_features(gemm_tiled_v1_lab PRIVATE cxx_std_20)
ai_system_enable_warnings(gemm_tiled_v1_lab)

if(AI_SYSTEM_ENABLE_TESTS)
    add_test(NAME gemm_tiled_v1_lab COMMAND gemm_tiled_v1_lab)
endif()
