add_library(ai_system_hgemm_lab)
target_sources(ai_system_hgemm_lab PRIVATE hgemm_lab.cpp)

if(AI_SYSTEM_ENABLE_CUDA)
    target_sources(ai_system_hgemm_lab PRIVATE hgemm_lab.cu hgemm_mma_basic.cu hgemm_wmma_basic.cu)
    ai_system_configure_cuda_target(ai_system_hgemm_lab)
endif()

target_include_directories(ai_system_hgemm_lab PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ai_system_hgemm_lab PUBLIC ai_system_core)
target_compile_features(ai_system_hgemm_lab PUBLIC cxx_std_20)
ai_system_enable_warnings(ai_system_hgemm_lab)

add_executable(hgemm_benchmark_lab hgemm_benchmark_lab.cpp)
target_link_libraries(hgemm_benchmark_lab PRIVATE ai_system_hgemm_lab)
target_compile_features(hgemm_benchmark_lab PRIVATE cxx_std_20)
ai_system_enable_warnings(hgemm_benchmark_lab)
