if(NOT AI_SYSTEM_ENABLE_CUDA)
    message(STATUS "Skipping labs/cute: AI_SYSTEM_ENABLE_CUDA=OFF")
    return()
endif()

set(AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT "${AI_SYSTEM_CUTLASS_ROOT}")
if(NOT AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT AND DEFINED ENV{CUTLASS_ROOT})
    set(AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT "$ENV{CUTLASS_ROOT}")
endif()

if(NOT AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT)
    message(STATUS "Skipping labs/cute examples: set AI_SYSTEM_CUTLASS_ROOT or CUTLASS_ROOT to a CUTLASS checkout")
    return()
endif()

if(NOT EXISTS "${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include/cute/tensor.hpp")
    message(STATUS "Skipping labs/cute examples: ${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include/cute/tensor.hpp not found")
    return()
endif()

add_library(ai_system_cute_headers INTERFACE)
target_include_directories(
    ai_system_cute_headers
    INTERFACE
        "${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include"
)
target_compile_features(ai_system_cute_headers INTERFACE cxx_std_17)

add_executable(cute_layout_mapping examples/cute_layout_mapping.cu)
target_link_libraries(cute_layout_mapping PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_layout_mapping PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_layout_mapping)

add_executable(cute_layout_algebra_demo examples/cute_layout_algebra_demo.cu)
target_link_libraries(cute_layout_algebra_demo PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_layout_algebra_demo PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_layout_algebra_demo)

add_executable(cute_tensor_tile_demo examples/cute_tensor_tile_demo.cu)
target_link_libraries(cute_tensor_tile_demo PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_tensor_tile_demo PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_tensor_tile_demo)

add_executable(cute_copy_g2s_naive examples/cute_copy_g2s_naive.cu)
target_link_libraries(cute_copy_g2s_naive PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_copy_g2s_naive PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_copy_g2s_naive)

add_executable(cute_copy_g2s_cpasync examples/cute_copy_g2s_cpasync.cu)
target_link_libraries(cute_copy_g2s_cpasync PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_copy_g2s_cpasync PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_copy_g2s_cpasync)

add_executable(cute_copy_s2r examples/cute_copy_s2r.cu)
target_link_libraries(cute_copy_s2r PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_copy_s2r PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_copy_s2r)

add_executable(cute_smem_swizzle_demo examples/cute_smem_swizzle_demo.cu)
target_link_libraries(cute_smem_swizzle_demo PRIVATE ai_system_cute_headers CUDA::cudart)
target_compile_features(cute_smem_swizzle_demo PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cute_smem_swizzle_demo)
