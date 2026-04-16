cmake_minimum_required(VERSION 3.22)

project(
    AI_system
    VERSION 0.1.0
    DESCRIPTION "A cross-platform AI system learning workspace for CUDA, compiler, and deployment labs."
    LANGUAGES CXX
)

include(CTest)

option(AI_SYSTEM_ENABLE_CUDA "Build CUDA kernels and runtime GPU utilities." ON)
option(AI_SYSTEM_BUILD_LABS "Build hands-on learning lab executables." ON)
option(AI_SYSTEM_ENABLE_TESTS "Build CTest smoke tests." ${BUILD_TESTING})

set(
    AI_SYSTEM_GPU_PROFILE
    "native"
    CACHE STRING
    "Target GPU profile: native, 4090, 5060, all, or a semicolon-separated SM list."
)
set_property(CACHE AI_SYSTEM_GPU_PROFILE PROPERTY STRINGS native 4090 5060 all)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(CudaArchitectures)

if(AI_SYSTEM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        ai_system_resolve_cuda_architectures(AI_SYSTEM_RESOLVED_CUDA_ARCHS AI_SYSTEM_RESOLVED_GPU_LABELS)
        set(CMAKE_CUDA_ARCHITECTURES "${AI_SYSTEM_RESOLVED_CUDA_ARCHS}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "AI_system CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} (${AI_SYSTEM_RESOLVED_GPU_LABELS})")
    else()
        message(WARNING "CUDA compiler not found. Building AI_system in CPU-only mode.")
        set(AI_SYSTEM_ENABLE_CUDA OFF)
    endif()
endif()

if(AI_SYSTEM_ENABLE_CUDA)
    set(AI_SYSTEM_HAS_CUDA_CONFIG 1)
    set(AI_SYSTEM_CUDA_ARCHS_STRING "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(AI_SYSTEM_HAS_CUDA_CONFIG 0)
    set(AI_SYSTEM_CUDA_ARCHS_STRING "cpu-only")
    set(AI_SYSTEM_RESOLVED_GPU_LABELS "CPU fallback")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ai_system/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include/ai_system/config.hpp"
    @ONLY
)

add_library(ai_system_core)
target_sources(
    ai_system_core
    PRIVATE
        src/benchmark/benchmark_report.cpp
        src/benchmark/benchmark_runner.cpp
        src/kernels/basic_kernels.cpp
        src/plan/learning_plan.cpp
        src/runtime/gpu_info.cpp
)

if(AI_SYSTEM_ENABLE_CUDA)
    target_sources(
        ai_system_core
        PRIVATE
            src/kernels/basic_kernels.cu
    )
endif()

target_include_directories(
    ai_system_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/include"
)

target_compile_features(ai_system_core PUBLIC cxx_std_20)

if(AI_SYSTEM_ENABLE_CUDA)
    target_link_libraries(ai_system_core PUBLIC CUDA::cudart)
    ai_system_configure_cuda_target(ai_system_core)
endif()

ai_system_enable_warnings(ai_system_core)

add_executable(ai_system_cli src/app/main.cpp)
target_link_libraries(ai_system_cli PRIVATE ai_system_core)
target_compile_features(ai_system_cli PRIVATE cxx_std_20)
ai_system_enable_warnings(ai_system_cli)

if(AI_SYSTEM_BUILD_LABS)
    add_subdirectory(labs/perf_engineering)
endif()

if(AI_SYSTEM_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
