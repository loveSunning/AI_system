# CuTe Scripts

Scripts are thin wrappers around the root CMake presets.

Windows:

```powershell
.\labs\cute\scripts\check_env.ps1
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1
```

Linux / WSL:

```bash
labs/cute/scripts/check_env.sh
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh
```

Profiles:

- Windows scripts target RTX 5060 / `sm_120`.
- Linux scripts target RTX 4090D / `sm_89`.
- The default CUTLASS root is `3rdparty/cutlass`.
