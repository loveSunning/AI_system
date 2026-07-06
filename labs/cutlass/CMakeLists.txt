if(NOT AI_SYSTEM_ENABLE_CUDA)
    message(STATUS "Skipping labs/cutlass: AI_SYSTEM_ENABLE_CUDA=OFF")
    return()
endif()

set(AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT "${AI_SYSTEM_CUTLASS_ROOT}")
if(NOT AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT AND DEFINED ENV{CUTLASS_ROOT})
    set(AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT "$ENV{CUTLASS_ROOT}")
endif()

if(NOT AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT)
    message(STATUS "Skipping labs/cutlass examples: set AI_SYSTEM_CUTLASS_ROOT or CUTLASS_ROOT to a CUTLASS checkout")
    return()
endif()

if(NOT EXISTS "${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include/cutlass/cutlass.h")
    message(STATUS "Skipping labs/cutlass examples: ${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include/cutlass/cutlass.h not found")
    return()
endif()

add_library(ai_system_cutlass_headers INTERFACE)
target_include_directories(
    ai_system_cutlass_headers
    INTERFACE
        "${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/include"
        "${AI_SYSTEM_EFFECTIVE_CUTLASS_ROOT}/tools/util/include"
)
target_compile_features(ai_system_cutlass_headers INTERFACE cxx_std_17)

add_executable(cutlass_header_probe examples/cutlass_header_probe.cu)
target_link_libraries(cutlass_header_probe PRIVATE ai_system_cutlass_headers CUDA::cudart)
target_compile_features(cutlass_header_probe PRIVATE cxx_std_17)
ai_system_configure_cuda_target(cutlass_header_probe)
