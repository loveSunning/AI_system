from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TRITON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRITON_ROOT.parents[1]
OUT_BENCHMARK_ROOT = REPO_ROOT / "out" / "triton" / "benchmarks"


def parse_shape_x(op: str, shape: str) -> int:
    if op == "vector_add":
        return int(shape)
    if op == "fused_softmax":
        _, cols = shape.lower().split("x", maxsplit=1)
        return int(cols)
    raise ValueError(f"unsupported op: {op}")


def label_for(row: dict[str, str]) -> str:
    impl = row["impl"]
    if impl == "triton":
        return f"{impl} {row['config']}"
    return impl


def load_rows(path: Path, op: str, dtype: str | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("op") == op]
    if dtype is not None:
        rows = [row for row in rows if row.get("dtype") == dtype]
    return rows


def reduce_points(rows: list[dict[str, str]], op: str, metric: str) -> dict[str, list[tuple[int, float]]]:
    best: dict[tuple[str, int], float] = {}
    for row in rows:
        x_value = parse_shape_x(op, row["shape"])
        label = label_for(row)
        y_value = float(row[metric])
        key = (label, x_value)
        if key not in best:
            best[key] = y_value
        elif metric == "throughput":
            best[key] = max(best[key], y_value)
        else:
            best[key] = min(best[key], y_value)

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (label, x_value), y_value in best.items():
        grouped[label].append((x_value, y_value))
    return {label: sorted(points) for label, points in grouped.items()}


def plot_metric(rows: list[dict[str, str]], op: str, metric: str, output_dir: Path, dtype: str | None) -> Path | None:
    if not rows:
        return None

    import matplotlib.pyplot as plt

    grouped = reduce_points(rows, op, metric)
    if not grouped:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, points in grouped.items():
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        ax.plot(x_vals, y_vals, marker="o", label=label)

    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("n_elements" if op == "vector_add" else "n_cols")
    ax.set_ylabel("GB/s" if metric == "throughput" else "ms")
    dtype_text = f" {dtype}" if dtype is not None else ""
    title_metric = "Throughput" if metric == "throughput" else "Latency"
    ax.set_title(f"W09 {op}{dtype_text} {title_metric}")
    ax.legend()
    fig.tight_layout()

    suffix = "throughput" if metric == "throughput" else "latency"
    dtype_suffix = dtype if dtype is not None else "all"
    output_path = output_dir / f"w09-{op}-{dtype_suffix}-{suffix}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot W09 benchmark CSV results.")
    parser.add_argument("--input", type=Path, default=OUT_BENCHMARK_ROOT / "w09_vector_add_softmax.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT_BENCHMARK_ROOT / "plots")
    parser.add_argument("--op", choices=["all", "vector_add", "fused_softmax"], default="all")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--metric", choices=["both", "throughput", "avg_ms"], default="both")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"benchmark CSV not found: {args.input}")

    ops = ["vector_add", "fused_softmax"] if args.op == "all" else [args.op]
    metrics = ["throughput", "avg_ms"] if args.metric == "both" else [args.metric]

    for op in ops:
        rows = load_rows(args.input, op, args.dtype)
        for metric in metrics:
            output_path = plot_metric(rows, op, metric, args.output_dir, args.dtype)
            if output_path is None:
                print(f"skip: no rows for op={op} dtype={args.dtype} metric={metric}")
            else:
                print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
