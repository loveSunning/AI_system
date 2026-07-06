# CUTLASS Scripts

These scripts keep the first CUTLASS stage reproducible on Windows and Linux.

## Windows

```powershell
.\labs\cutlass\scripts\check_env.ps1 -CutlassRoot D:\deps\cutlass
.\labs\cutlass\scripts\configure.ps1 -CutlassRoot D:\deps\cutlass
.\labs\cutlass\scripts\build.ps1
```

## Linux

```bash
./labs/cutlass/scripts/check_env.sh /opt/cutlass
./labs/cutlass/scripts/configure.sh --cutlass-root /opt/cutlass
./labs/cutlass/scripts/build.sh
```

Windows scripts target RTX 5060 / `sm_120`; Linux scripts target RTX 4090D / `sm_89`.

## Profiler

The profiler wrapper expects a CUTLASS build tree that already contains the official
`cutlass_profiler` binary.

```powershell
.\labs\cutlass\scripts\run_profiler.ps1 -ProfilerPath D:\deps\cutlass\build\tools\profiler\Release\cutlass_profiler.exe
```

```bash
./labs/cutlass/scripts/run_profiler.sh --profiler /opt/cutlass/build/tools/profiler/cutlass_profiler
```
