# CUTLASS Scripts

These scripts are optional wrappers. Direct CMake commands are documented in
`labs/cutlass/README.md`.

## Windows

```powershell
.\labs\cutlass\scripts\check_env.ps1
.\labs\cutlass\scripts\configure.ps1
.\labs\cutlass\scripts\build.ps1
```

## Linux

```bash
bash ./labs/cutlass/scripts/check_env.sh
bash ./labs/cutlass/scripts/configure.sh
bash ./labs/cutlass/scripts/build.sh
```

Windows scripts target RTX 5060 / `sm_120`; Linux scripts target RTX 4090D / `sm_89`.
The default CUTLASS root is `3rdparty/cutlass`.

## Profiler

Profiler is optional and requires configuring and building the official CUTLASS
build tree first.

```powershell
.\labs\cutlass\scripts\configure_official_cutlass.ps1
.\labs\cutlass\scripts\build_official_cutlass.ps1
.\labs\cutlass\scripts\run_profiler.ps1
```

```bash
bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
bash ./labs/cutlass/scripts/run_profiler.sh
```
