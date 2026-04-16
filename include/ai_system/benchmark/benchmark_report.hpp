#pragma once

#include "ai_system/benchmark/benchmark_runner.hpp"

#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace ai_system::benchmark {

struct BenchmarkRow {
    std::string op;
    std::string impl;
    std::string shape;
    BenchmarkResult result;
    std::optional<double> perf_value;
    std::string perf_unit;
};

struct ValidationRow {
    std::string op;
    std::string impl;
    std::string check;
    std::string status;
    std::string detail;
};

struct BenchmarkReport {
    std::vector<BenchmarkRow> benchmark_rows;
    std::vector<ValidationRow> validation_rows;
};

std::string format_decimal(double value, int precision = 3);

void add_benchmark_row(
    BenchmarkReport& report,
    std::string op,
    std::string impl,
    std::string shape,
    const BenchmarkResult& result,
    std::optional<double> perf_value,
    std::string perf_unit
);

void add_validation_row(
    BenchmarkReport& report,
    std::string op,
    std::string impl,
    std::string check,
    std::string status,
    std::string detail = {}
);

void print_benchmark_table(const BenchmarkReport& report, std::ostream& output = std::cout);
void print_validation_table(const BenchmarkReport& report, std::ostream& output = std::cout);

}  // namespace ai_system::benchmark
