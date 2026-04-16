#include "ai_system/benchmark/benchmark_report.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ai_system::benchmark {

namespace {

struct TableColumn {
    std::string header;
    bool right_align {false};
};

std::string sanitize_cell(std::string value) {
    for(char& ch : value) {
        if(ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }
    return value.empty() ? "-" : value;
}

void print_horizontal_rule(std::ostream& output, const std::vector<std::size_t>& widths) {
    output << '+';
    for(const std::size_t width : widths) {
        output << std::string(width + 2, '-') << '+';
    }
    output << "\n";
}

void print_table_row(
    std::ostream& output,
    const std::vector<std::string>& row,
    const std::vector<std::size_t>& widths,
    const std::vector<TableColumn>& columns
) {
    output << '|';
    for(std::size_t index = 0; index < row.size(); ++index) {
        const auto cell = sanitize_cell(row[index]);
        output << ' ';
        if(columns[index].right_align) {
            output << std::right << std::setw(static_cast<int>(widths[index])) << cell;
        } else {
            output << std::left << std::setw(static_cast<int>(widths[index])) << cell;
        }
        output << ' ' << '|';
    }
    output << "\n";
}

void print_table(
    std::ostream& output,
    const std::vector<TableColumn>& columns,
    const std::vector<std::vector<std::string>>& rows
) {
    std::vector<std::size_t> widths;
    widths.reserve(columns.size());
    for(const auto& column : columns) {
        widths.push_back(column.header.size());
    }

    for(const auto& row : rows) {
        for(std::size_t index = 0; index < row.size(); ++index) {
            widths[index] = std::max(widths[index], sanitize_cell(row[index]).size());
        }
    }

    std::vector<std::string> header_cells;
    header_cells.reserve(columns.size());
    for(const auto& column : columns) {
        header_cells.push_back(column.header);
    }

    print_horizontal_rule(output, widths);
    print_table_row(output, header_cells, widths, columns);
    print_horizontal_rule(output, widths);
    for(const auto& row : rows) {
        print_table_row(output, row, widths, columns);
    }
    print_horizontal_rule(output, widths);
}

}  // namespace

std::string format_decimal(double value, int precision) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

void add_benchmark_row(
    BenchmarkReport& report,
    std::string op,
    std::string impl,
    std::string shape,
    const BenchmarkResult& result,
    std::optional<double> perf_value,
    std::string perf_unit
) {
    report.benchmark_rows.push_back(BenchmarkRow {
        std::move(op),
        std::move(impl),
        std::move(shape),
        result,
        perf_value,
        std::move(perf_unit)
    });
}

void add_validation_row(
    BenchmarkReport& report,
    std::string op,
    std::string impl,
    std::string check,
    std::string status,
    std::string detail
) {
    report.validation_rows.push_back(ValidationRow {
        std::move(op),
        std::move(impl),
        std::move(check),
        std::move(status),
        std::move(detail)
    });
}

void print_benchmark_table(const BenchmarkReport& report, std::ostream& output) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(report.benchmark_rows.size());

    for(const auto& row : report.benchmark_rows) {
        rows.push_back({
            row.op,
            row.impl,
            row.shape,
            format_decimal(row.result.average_ms),
            format_decimal(row.result.min_ms),
            format_decimal(row.result.max_ms),
            row.perf_value.has_value() ? format_decimal(*row.perf_value) : std::string("n/a"),
            row.perf_unit.empty() ? std::string("-") : row.perf_unit,
            std::to_string(row.result.warmup_iterations),
            std::to_string(row.result.measured_iterations)
        });
    }

    output << "\nBenchmark Results\n";
    print_table(
        output,
        {
            {"op"},
            {"impl"},
            {"shape"},
            {"avg_ms", true},
            {"min_ms", true},
            {"max_ms", true},
            {"perf", true},
            {"unit"},
            {"warmup", true},
            {"iters", true}
        },
        rows
    );
}

void print_validation_table(const BenchmarkReport& report, std::ostream& output) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(report.validation_rows.size());

    for(const auto& row : report.validation_rows) {
        rows.push_back({
            row.op,
            row.impl,
            row.check,
            row.status,
            row.detail
        });
    }

    output << "\nValidation\n";
    print_table(
        output,
        {
            {"op"},
            {"impl"},
            {"check"},
            {"status"},
            {"detail"}
        },
        rows
    );
}

}  // namespace ai_system::benchmark
