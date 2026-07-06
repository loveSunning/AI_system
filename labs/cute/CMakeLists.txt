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
