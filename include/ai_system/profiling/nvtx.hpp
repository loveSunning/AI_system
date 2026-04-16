#pragma once

#include "ai_system/config.hpp"

#include <string>
#include <string_view>

#if AI_SYSTEM_HAS_CUDA
#    if defined(__has_include)
#        if __has_include(<nvtx3/nvToolsExt.h>)
#            include <nvtx3/nvToolsExt.h>
#            define AI_SYSTEM_NVTX_HEADER_AVAILABLE 1
#        else
#            define AI_SYSTEM_NVTX_HEADER_AVAILABLE 0
#        endif
#    else
#        define AI_SYSTEM_NVTX_HEADER_AVAILABLE 0
#    endif
#else
#    define AI_SYSTEM_NVTX_HEADER_AVAILABLE 0
#endif

namespace ai_system::profiling {

inline constexpr bool kNvtxAvailable = AI_SYSTEM_NVTX_HEADER_AVAILABLE != 0;

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name) noexcept {
        active_ = push(name);
    }

    explicit ScopedNvtxRange(std::string_view name)
        : owned_name_(name) {
        active_ = push(owned_name_.c_str());
    }

    ~ScopedNvtxRange() noexcept {
        if(active_) {
            pop();
        }
    }

    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange(ScopedNvtxRange&&) = delete;
    ScopedNvtxRange& operator=(ScopedNvtxRange&&) = delete;

private:
    static bool push(const char* name) noexcept {
#if AI_SYSTEM_NVTX_HEADER_AVAILABLE
        if(name == nullptr || *name == '\0') {
            return false;
        }
        nvtxRangePushA(name);
        return true;
#else
        (void)name;
        return false;
#endif
    }

    static void pop() noexcept {
#if AI_SYSTEM_NVTX_HEADER_AVAILABLE
        nvtxRangePop();
#endif
    }

    std::string owned_name_;
    bool active_ {false};
};

}  // namespace ai_system::profiling
