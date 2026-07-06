# CUTLASS Scripts

These scripts keep the first CUTLASS stage reproducible on Windows and Linux.

## Windows

```powershell
.\labs\cutlass\scripts\check_env.ps1
.\labs\cutlass\scripts\configure.ps1
.\labs\cutlass\scripts\build.ps1
.\labs\cutlass\scripts\configure_official_cutlass.ps1
```

## Linux

```bash
./labs/cutlass/scripts/check_env.sh
./labs/cutlass/scripts/configure.sh
./labs/cutlass/scripts/build.sh
./labs/cutlass/scripts/configure_official_cutlass.sh
```

Windows scripts target RTX 5060 / `sm_120`; Linux scripts target RTX 4090D / `sm_89`.
The default CUTLASS root is `3rdparty/cutlass`.

## Profiler

The profiler wrapper expects a CUTLASS build tree that already contains the official
`cutlass_profiler` binary.

```powershell
.\labs\cutlass\scripts\run_profiler.ps1 -ProfilerPath .\3rdparty\cutlass\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe
```

```bash
./labs/cutlass/scripts/run_profiler.sh --profiler ./3rdparty/cutlass/build/linux-4090d/tools/profiler/cutlass_profiler
```
