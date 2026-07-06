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
.\labs\cutlass\scripts\run_profiler.ps1 -ProfilerPath .\3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe -M 4096 -N 4096 -K 4096
```

```bash
labs/cutlass/scripts/run_profiler.sh --profiler ./3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler --m 4096 --n 4096 --k 4096
```
