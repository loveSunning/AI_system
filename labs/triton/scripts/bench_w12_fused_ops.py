from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRITON_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = TRITON_ROOT / "python"


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=TRITON_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the W12 fused-ops benchmark set: Dropout, LayerNorm, and RMSNorm.")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--mode", choices=["forward", "backward"], default="backward")
    parser.add_argument("--dropout-elements", type=int, default=1 << 24)
    parser.add_argument("--dropout-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sweep", action="store_true", help="Run the default sweep for each W12 op.")
    parser.add_argument("--plot", action="store_true", help="Generate Triton perf_report plots for sweep mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    python = sys.executable
    commands = [
        [
            python,
            "scripts/bench_dropout.py",
            "--n-elements",
            str(args.dropout_elements),
            "--dtype",
            args.dropout_dtype,
            "--p",
            str(args.dropout_p),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
        [
            python,
            "scripts/bench_layer_norm.py",
            "--rows",
            str(args.rows),
            "--cols",
            str(args.cols),
            "--dtype",
            args.dtype,
            "--mode",
            args.mode,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
        [
            python,
            "scripts/bench_rms_norm.py",
            "--rows",
            str(args.rows),
            "--cols",
            str(args.cols),
            "--dtype",
            args.dtype,
            "--mode",
            args.mode,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
    ]

    if args.sweep:
        for command in commands:
            command.append("--sweep")
    if args.plot:
        for command in commands:
            command.append("--plot")

    pythonpath = str(PYTHON_ROOT)
    if pythonpath not in sys.path:
        sys.path.insert(0, pythonpath)

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
