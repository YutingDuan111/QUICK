# pyquick — Build & Install Guide

pyquick is QUICK's Python API (`PyQuick`/`Job` classes, via f2py). One source tree builds
a serial CPU, an MPI (CPU), a GPU or a GPU + MPI (one GPU per rank) extension — the flavour
is a **compile-time** choice; nothing changes in your Python code afterward, only which install
you `source` before `import pyquick`. `-DCUDA=TRUE -DMPI=TRUE` gives GPU + MPI; otherwise GPU
wins over MPI, and MPI over serial. `pyquick.gpu_enabled()` and `pyquick.mpi_enabled()` tell
which one you imported.

## Prerequisites

- CMake >= 3.12, a Fortran/C/C++ compiler (`-DCOMPILER=GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`,
  `PGI`, or `AUTO` — see `AGENTS.md`)
- Python 3.10+ **with development headers**. A venv from a headerless system Python will not
  work (meson fails with "Python dependency not found") — use an interpreter that ships its own
  headers + pkg-config: conda/miniforge, pyenv, or a system `python3-dev`/`python3-devel` package.
- For that same Python: `numpy >= 2.0`, `meson`, `ninja`. numpy 1.26's f2py silently drops
  `--f90flags`, so the `#if defined(GPU)` guards are never preprocessed and the extension fails
  to import (configure now stops with an error on numpy < 2.0):
  ```bash
  python3 -m pip install "numpy>=2.0" meson ninja
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

### MPI (CPU)

One calculation runs on all MPI ranks (QUICK's `libquick_mpi`). Python owns MPI through
**mpi4py**, which must be built against the **same MPI library** as QUICK — install it from
source with that MPI's `mpicc` (reinstall it after upgrading the MPI library):

```bash
MPICC=$(which mpicc) python3 -m pip install --no-binary=mpi4py mpi4py
python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # the MPI QUICK uses

mkdir build-mpi && cd build-mpi
cmake .. -DCOMPILER=GNU -DMPI=TRUE -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install-mpi
# configure must print: "pyquick: binding against MPI libquick (quick_mpi)"
cmake --build . --parallel $(nproc) && cmake --install .
source ../install-mpi/quick.rc
```

Run any pyquick script under `mpiexec`, always through `python -m mpi4py` (an uncaught
exception then aborts every rank instead of leaving the others waiting):

```bash
mpiexec -n 4 python3 -m mpi4py my_script.py
```

- Every rank runs the whole script and gets the **same** `Result`; only rank 0 writes
  `<name>.out`. Print from one rank: `if pyquick.mpi_rank() == 0: print(...)`.
  `pyquick.mpi_enabled()`, `mpi_rank()`, `mpi_size()` report the build and the ranks
  (`False`, `0`, `1` on serial/GPU builds).
- A calculation always uses every rank of `MPI_COMM_WORLD` (QUICK hard-codes it), so all ranks
  must make the same pyquick calls in the same order. Do not compute different molecules on
  different ranks with this build — use task farming (below) for that.
- An error inside QUICK aborts **all** ranks (`MPI_Abort`); errors before QUICK starts (bad
  geometry, settings) still raise normally on every rank.
- `python script.py` without `mpiexec` runs as a single rank. Jupyter is not supported for MPI
  runs, and a subprocess started from an MPI rank must not inherit its `OMPI_*`/`PMIX_*`
  environment variables (it would join the job as a duplicate rank).

**Task farming** (many independent calculations, one per rank) needs the **serial** build plus
mpi4py, not the MPI build: see `src/pyquick/test/mpi_task_farm_example.py`
(`mpiexec -n 4 python3 -m mpi4py mpi_task_farm_example.py`).

### GPU + MPI (one GPU per rank)

QUICK's multi-GPU mode (as in `quick.cuda.MPI`): each MPI rank drives its own GPU, and the
partial Fock matrices are summed on the host, so any MPI works (no CUDA-aware MPI needed).
Install mpi4py exactly as for MPI (CPU), against the MPI QUICK uses, then:

```bash
mkdir build-gpumpi && cd build-gpumpi
cmake .. -DCOMPILER=GNU -DCUDA=TRUE -DMPI=TRUE -DQUICK_USER_ARCH=<arch> -DPYTHON=TRUE \
    -DPython3_EXECUTABLE=$(which python3) -DCMAKE_INSTALL_PREFIX=$PWD/../install-gpumpi
# configure must print: "pyquick: binding against multi-GPU MPI libquick (quick_mpi_cuda)"
cmake --build . --parallel $(nproc) && cmake --install .
source ../install-gpumpi/quick.rc
```

Run with **at most one rank per visible GPU**; pick the GPUs with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python3 -m mpi4py my_script.py
```

- Rank r uses visible GPU `r % count`. More ranks than GPUs stops every rank with
  `more MPI ranks (N) than visible GPUs` (QUICK's own limit).
- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`: by default CUDA numbers GPUs fastest-first, so on a
  machine with mixed GPUs `CUDA_VISIBLE_DEVICES=0,1` may not be the cards `nvidia-smi` calls
  0 and 1 (and a build for one architecture cannot run on another).
- Use GPUs of one architecture, matching `-DQUICK_USER_ARCH`.
- Check every rank got its own GPU: `grep -A3 "MPI RANK" <name>.out` lists one device per rank.
- Everything from MPI (CPU) applies (same `Result` on every rank, rank 0 writes the log, an
  engine error aborts all ranks), and so does the GPU known issue: persistent jobs leak a
  little device memory per step.

## Verify

```bash
python3 -c "
import pyquick
r = pyquick.PyQuick(method='HF', basis='STO-3G').get_energy(
    'O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24', name='verify')
print(r.total_energy)
"
grep -i 'gpu\|device\|cuda' verify.out   # GPU builds: confirms the device was actually used
```

Or run the test suite: `pytest src/pyquick/test/test_pyquick_api.py -q`.

MPI build (from `src/pyquick/test/`):

```bash
mpiexec -n 4 python3 -m mpi4py -c "import pyquick; print(pyquick.mpi_rank(), pyquick.mpi_size())"
mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_api.py -q   # same suite, on every rank
mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_mpi.py -q   # MPI-only checks (also -n 4)
python3 -m pytest test_pyquick_mpi_launch.py -q                    # starts mpiexec itself
mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_gpu_mpi.py -q  # GPU + MPI build only
grep "TOTAL RANKS" <name>.out                                      # QUICK ran on all ranks
```

`test_pyquick_mpi_launch.py` reads two optional variables: `PYQUICK_MPIEXEC` (e.g.
`"mpiexec --oversubscribe"` when ranks > cores) and `PYQUICK_SERIAL_LIB` (a serial install's
`lib` directory, enabling the task-farming test).

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