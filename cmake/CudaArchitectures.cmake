function(ai_system__compute_capability_to_arch out_var capability)
    string(REPLACE "." "" architecture "${capability}")
    set(${out_var} "${architecture}" PARENT_SCOPE)
endfunction()

function(ai_system_detect_native_cuda_architectures out_arches out_labels)
    find_program(AI_SYSTEM_NVIDIA_SMI_EXECUTABLE NAMES nvidia-smi)
    if(NOT AI_SYSTEM_NVIDIA_SMI_EXECUTABLE)
        set(${out_arches} "" PARENT_SCOPE)
        set(${out_labels} "" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND "${AI_SYSTEM_NVIDIA_SMI_EXECUTABLE}" "--query-gpu=name,compute_cap" "--format=csv,noheader"
        OUTPUT_VARIABLE raw_gpu_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(NOT raw_gpu_output)
        set(${out_arches} "" PARENT_SCOPE)
        set(${out_labels} "" PARENT_SCOPE)
        return()
    endif()

    string(REPLACE "\r" "" raw_gpu_output "${raw_gpu_output}")
    string(REPLACE "\n" ";" gpu_lines "${raw_gpu_output}")

    set(architectures "")
    set(labels "")

    foreach(line IN LISTS gpu_lines)
        if(line STREQUAL "")
            continue()
        endif()

        string(REGEX MATCH "^(.*),[ ]*([0-9]+\\.[0-9]+)$" _ "${line}")
        if(CMAKE_MATCH_2)
            string(STRIP "${CMAKE_MATCH_1}" gpu_name)
            ai_system__compute_capability_to_arch(gpu_arch "${CMAKE_MATCH_2}")
            list(APPEND architectures "${gpu_arch}")
            list(APPEND labels "${gpu_name}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES architectures)
    list(REMOVE_DUPLICATES labels)
    list(JOIN architectures ";" joined_architectures)
    list(JOIN labels ", " joined_labels)

    set(${out_arches} "${joined_architectures}" PARENT_SCOPE)
    set(${out_labels} "${joined_labels}" PARENT_SCOPE)
endfunction()

function(ai_system_resolve_cuda_architectures out_arches out_labels)
    string(TOLOWER "${AI_SYSTEM_GPU_PROFILE}" normalized_profile)

    if(normalized_profile STREQUAL "native")
        ai_system_detect_native_cuda_architectures(resolved_architectures resolved_labels)
        if(NOT resolved_architectures)
            if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
                set(resolved_architectures "120")
                set(resolved_labels "Fallback: Windows RTX 5060")
            elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
                set(resolved_architectures "89")
                set(resolved_labels "Fallback: Linux RTX 4090D")
            else()
                set(resolved_architectures "89")
                set(resolved_labels "Fallback: RTX 4090D")
            endif()
        endif()
    elseif(normalized_profile STREQUAL "4090d" OR
           normalized_profile STREQUAL "rtx4090d")
        set(resolved_architectures "89")
        set(resolved_labels "RTX 4090D (Ada Lovelace)")
    elseif(normalized_profile STREQUAL "5060" OR normalized_profile STREQUAL "rtx5060")
        set(resolved_architectures "120")
        set(resolved_labels "RTX 5060 (Blackwell)")
    elseif(normalized_profile MATCHES "^[0-9]+(;[0-9]+)*$")
        set(resolved_architectures "${AI_SYSTEM_GPU_PROFILE}")
        set(resolved_labels "Custom SM list")
    else()
        message(
            FATAL_ERROR
            "Unsupported AI_SYSTEM_GPU_PROFILE='${AI_SYSTEM_GPU_PROFILE}'. "
            "Use native, 4090d, 5060, or a semicolon-separated SM list such as 89;120."
        )
    endif()

    set(${out_arches} "${resolved_architectures}" PARENT_SCOPE)
    set(${out_labels} "${resolved_labels}" PARENT_SCOPE)
endfunction()
