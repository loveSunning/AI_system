# CUTLASS Benchmarks

Raw profiler CSV and generated summaries should go under:

```text
out/cutlass/
```

Planned benchmark artifacts:

- `out/cutlass/profiler_smoke.csv`
- `out/cutlass/profiler_sweep.csv`
- `out/cutlass/epilogue_sweep.csv`

Start with profiler smoke:

```powershell
.\labs\cutlass\scripts\configure_official_cutlass.ps1
.\labs\cutlass\scripts\build_official_cutlass.ps1
.\labs\cutlass\scripts\run_profiler.ps1 -M 4096 -N 4096 -K 4096
```

```bash
bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
bash ./labs/cutlass/scripts/run_profiler.sh --m 4096 --n 4096 --k 4096
```
