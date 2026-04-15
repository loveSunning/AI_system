#pragma once

#include <string>
#include <vector>

namespace ai_system::plan {

struct LearningPhase {
    std::string month;
    std::string weeks;
    std::string topic;
    std::string directory;
    std::string deliverable;
};

const std::vector<LearningPhase>& learning_plan();

}  // namespace ai_system::plan
