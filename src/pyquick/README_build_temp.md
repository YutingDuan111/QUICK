# pyquick — Build & Install Guide

pyquick is QUICK's Python API (`Calculation`/`Job` classes, via f2py). One source tree builds
either a CPU or a GPU extension — GPU is a **compile-time** choice; nothing changes in your
Python code afterward, only which install you `source` before `import pyquick`.

## Prerequisites

- CMake >= 3.12, a Fortran/C/C++ compiler (`-DCOMPILER=GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`,
  `PGI`, or `AUTO` — see `AGENTS.md`)
- Python 3.10+ **with development headers**. A venv from a headerless system Python will not
  work (meson fails with "Python dependency not found") — use an interpreter that ships its own
  headers + pkg-config: conda/miniforge, pyenv, or a system `python3-dev`/`python3-devel` package.
- For that same Python: `numpy >= 1.26` (older versions lack f2py's `--backend meson`), `meson`,
  `ninja`:
  ```bash
  python3 -m pip install "numpy>=1.26" meson ninja
  ```
- GPU only: **NVIDIA** — CUDA Toolkit + a GNU/Clang version in the compatibility table in
  `quick-cmake/QUICKCudaConfig.cmake`. **AMD** — ROCm + `hipcc`.

## Build

For GPU, build in two passes: first with `-DPYTHON=FALSE` to confirm the CUDA/ROCm toolchain
and the plain `quick` binary work, *then* turn on Python. This isolates a toolchain failure from
an f2py failure.

### CPU

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc)
cmake --install .
source ../install/quick.rc
```

### GPU — NVIDIA (CUDA)

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DCOMPILER=GNU -DCUDA=TRUE -DQUICK_USER_ARCH=<arch> -DPYTHON=FALSE \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install-cuda
cmake --build . --parallel $(nproc) && cmake --install .
source ../install-cuda/quick.rc
quick.cuda ../test/ene_H2O_rhf_sto3g.in   # sanity check: no Python involved yet
grep -i 'gpu\|device\|cuda' ../test/ene_H2O_rhf_sto3g.out   # confirms the device was used

cmake .. -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3)
cmake --build . --parallel $(nproc) && cmake --install .
# configure must print: "pyquick: binding against GPU libquick (quick_cuda)"
```

`<arch>`: `kepler`, `maxwell`, `pascal`, `volta`, `turing`, `ampere`, `adalovelace`, `hopper`,
`blackwell`, `blackwell2` — pick the one matching your GPU, or omit `-DQUICK_USER_ARCH` to build
for several architectures based on the CUDA toolkit version.

### GPU — AMD (HIP)

Same recipe, `-DHIP=TRUE` instead of `-DCUDA=TRUE`; the plain-binary sanity check binary is also
named `quick.<target>` (target name follows `QUICK_GPU_TARGET_NAME`, i.e. `quick.hip`):

```bash
cmake .. -DCOMPILER=GNU -DHIP=TRUE -DQUICK_USER_ARCH=<arch> -DPYTHON=FALSE \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install-hip
cmake --build . --parallel $(nproc) && cmake --install .
source ../install-hip/quick.rc
quick.hip ../test/ene_H2O_rhf_sto3g.in
# then -DPYTHON=TRUE, same as the CUDA pass above
```

`<arch>`: `gfx908`, `gfx90a`, `gfx942`.

> **Status**: the HIP build path unverified on real hardware. 

## Verify

```bash
python3 -c "
import pyquick
r = pyquick.Calculation(method='HF', basis='STO-3G').get_energy(
    'O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24', name='verify')
print(r.total_energy)
"
grep -i 'gpu\|device\|cuda' verify.out   # GPU builds: confirms the device was actually used
```

Or run the test suite: `pytest src/pyquick/test/test_pyquick_api.py -q`.

## Common pitfalls

- **"Python dependency not found" in meson** — your Python has no dev headers (see Prerequisites).
- **`meson`/`ninja` "not found" even though installed** — `src/pyquick/CMakeLists.txt` finds them
  via `PATH`, not relative to `-DPython3_EXECUTABLE`. Make sure that environment's `bin/` is on
  `PATH` at **both** configure and build time.
- **GPU build "succeeds" but the `.so` never touches the device** — f2py's meson backend has
  historically dropped `--f90flags` on some versions, silently compiling the `#if defined(GPU)`
  guards out. Verify: `nm -D <install>/lib/pyquick/_pyquick*.so | grep gpu_` should show undefined
  `gpu_*_` symbols. If it shows none, the flags were dropped and you built a CPU-only `.so` that
  looks like a GPU one.
- **One Python version at a time** — a stale `_pyquick*.so` built for a different Python version
  silently shadows a fresh build if left on `PYTHONPATH`; rebuilding does not remove the old one
  from a different install prefix.