# pyquick — User and Build Guide

pyquick is QUICK's Python API (`PyQuick`, `Result` and `Job`, built with f2py). One QUICK
install holds up to four pyquick **engines**, and the import you write picks one:

| Mode | Import | Launch | The build needs |
|---|---|---|---|
| Serial | `import pyquick` | `python3 script.py` | nothing extra |
| MPI (CPU) | `import pyquick.mpi as pyquick` | `mpiexec -n 4 python3 -m mpi4py script.py` | `-DMPI=TRUE` |
| GPU | `import pyquick.gpu as pyquick` | `python3 script.py` | `-DCUDA=TRUE` (or `-DHIP=TRUE`) |
| GPU + MPI | `import pyquick.gpu_mpi as pyquick` | `mpiexec -n 2 python3 -m mpi4py script.py` | both |

Every mode has the same API, so switching modes changes one line of your script.

## Quick start

```python
import pyquick

calc = pyquick.PyQuick(method='B3LYP', basis='6-31G*')
result = calc.get_energy("""
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
""", name='water')                       # QUICK's log goes to water.out
print(result.total_energy)

result = calc.get_grad('mol.xyz')         # geometries can also come from an .xyz file
```

`get_energy`, `get_grad`, `geo_opt` and `new_job` all take a `'SYMBOL X Y Z'` string
(Angstrom) or the path of an `.xyz` file. `pyquick.read_xyz(path)` returns a file's geometry
as a string.

## Choosing a mode

- **One engine per process.** QUICK keeps its state in Fortran module variables, so a script
  uses one mode for its whole run. Importing a second one raises
  `ImportError: pyquick is already using the mpi engine in this process`.
- **A mode the install was built without** raises, for example,
  `ImportError: pyquick.gpu is not available: this QUICK install was built without GPU support`.
- **Plain `import pyquick` is serial**, and loads its engine on first use. `pyquick.read_xyz`
  works without loading any engine.
- **Plain `import pyquick` refuses to run under `mpiexec`** with more than one rank. An MPI
  script that forgot `pyquick.mpi` would otherwise run the whole calculation once on every
  rank, each on one core. The error names `pyquick.mpi` and `pyquick.gpu_mpi`.
- **`import pyquick.serial`** is the same serial engine without that check. Use it for task
  farming, where each rank runs its own independent calculations: see
  `src/pyquick/test/mpi_task_farm_example.py`.
- **Which engine am I on?** `pyquick.mpi_enabled()`, `gpu_enabled()`, `mpi_rank()` and
  `mpi_size()` report it (`False`, `False`, `0`, `1` for serial).

## Prerequisites

- CMake >= 3.12, a Fortran/C/C++ compiler (`-DCOMPILER=GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`,
  `PGI`, or `AUTO` — see `AGENTS.md`)
- Python 3.10+ **with development headers**. A venv from a headerless system Python will not
  work (meson fails with "Python dependency not found") — use an interpreter that ships its own
  headers + pkg-config: conda/miniforge, pyenv, or a system `python3-dev`/`python3-devel` package.
- For that same Python: `numpy >= 2.0`, `meson`, `ninja`. numpy 1.26's f2py silently drops
  `--f90flags`, so the `#if defined(GPU)` guards are never preprocessed and the extension fails
  to import (configure stops with an error on numpy < 2.0):
  ```bash
  python3 -m pip install "numpy>=2.0" meson ninja
  ```
- MPI modes: **mpi4py built against the same MPI library as QUICK**. Install it from source
  with that MPI's `mpicc`, and reinstall it after upgrading the MPI library:
  ```bash
  MPICC=$(which mpicc) python3 -m pip install --no-binary=mpi4py mpi4py
  python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # the MPI QUICK uses
  ```
- GPU modes: **NVIDIA** — CUDA Toolkit + a GNU/Clang version in the compatibility table in
  `quick-cmake/QUICKCudaConfig.cmake`. **AMD** — ROCm + `hipcc`.

## Build

One build makes every engine its flags allow. The serial engine is always built; `-DMPI=TRUE`
adds `mpi`, `-DCUDA=TRUE` (or `-DHIP=TRUE`) adds `gpu`, and both add `gpu_mpi` too.

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DMPI=TRUE -DCUDA=TRUE -DQUICK_USER_ARCH=<arch> \
    -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc)
cmake --install .
source ../install/quick.rc
```

Configure prints one line per engine, for example:

```
-- pyquick: serial engine, bound to libquick
-- pyquick: mpi engine, bound to libquick_mpi
-- pyquick: gpu engine, bound to libquick_cuda
-- pyquick: gpu_mpi engine, bound to libquick_mpi_cuda
```

Leave out `-DMPI=TRUE` or `-DCUDA=TRUE` for the modes you don't need. The engines are installed
in `<install>/lib/pyquick/_engines/<mode>/`.

### GPU builds

Build in two passes the first time: with `-DPYTHON=FALSE` to confirm the CUDA/ROCm toolchain and
the plain `quick` binary work, *then* turn on Python. This separates a toolchain failure from an
f2py failure.

```bash
cmake .. -DCOMPILER=GNU -DCUDA=TRUE -DQUICK_USER_ARCH=<arch> -DPYTHON=FALSE \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc) && cmake --install .
source ../install/quick.rc
quick.cuda ../test/ene_H2O_rhf_sto3g.in   # sanity check: no Python involved yet
grep -i 'gpu\|device\|cuda' ../test/ene_H2O_rhf_sto3g.out   # confirms the device was used

cmake .. -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3)
cmake --build . --parallel $(nproc) && cmake --install .
```

`<arch>` for NVIDIA: `kepler`, `maxwell`, `pascal`, `volta`, `turing`, `ampere`,
`adalovelace`, `hopper`, `blackwell`, `blackwell2` — the one matching your GPU, or omit
`-DQUICK_USER_ARCH` to build for several architectures based on the CUDA toolkit version.

AMD: the same recipe with `-DHIP=TRUE`, and `quick.hip` for the sanity check. `<arch>`:
`gfx908`, `gfx90a`, `gfx942`. The engines are still called `gpu` and `gpu_mpi`.
**Status:** the HIP path is unverified on real hardware.

## Running with MPI

Run under `mpiexec`, always through `python3 -m mpi4py` (an uncaught exception then aborts every
rank instead of leaving the others waiting):

```bash
mpiexec -n 4 python3 -m mpi4py my_script.py
```

- Every rank runs the whole script and gets the **same** `Result`; only rank 0 writes
  `<name>.out`. Print from one rank: `if pyquick.mpi_rank() == 0: print(...)`.
- A calculation always uses every rank of `MPI_COMM_WORLD`, so all ranks must make the same
  pyquick calls in the same order. For different molecules on different ranks, use task
  farming with `pyquick.serial` (see Choosing a mode).
- An error inside QUICK aborts **all** ranks (`MPI_Abort`); errors before QUICK starts (bad
  geometry, settings) still raise normally on every rank.
- `python3 script.py` without `mpiexec` runs as a single rank. Jupyter is not supported for MPI
  runs, and a subprocess started from an MPI rank must not inherit its `OMPI_*`/`PMIX_*`
  environment variables (it would join the job as a duplicate rank).
- Across several nodes, Open MPI does not forward environment variables by itself: pass the
  ones your script needs with `mpiexec -x`.

## Running on GPUs

`pyquick.gpu` uses one GPU; pick it with `CUDA_VISIBLE_DEVICES`. `pyquick.gpu_mpi` is QUICK's
multi-GPU mode (as in `quick.cuda.MPI`): each MPI rank drives a GPU, and the partial Fock
matrices are summed on the host, so any MPI works (no CUDA-aware MPI needed).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python3 -m mpi4py my_script.py
```

- Rank r uses visible GPU `r % count`, so with more ranks than GPUs some ranks share a GPU.
  One rank per GPU is fastest.
- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`: by default CUDA numbers GPUs fastest-first, so on a
  machine with mixed GPUs `CUDA_VISIBLE_DEVICES=0,1` may not be the cards `nvidia-smi` calls
  0 and 1.
- Use GPUs of one architecture, matching `-DQUICK_USER_ARCH`; a build for one architecture
  cannot run on another.
- Check every rank got its own GPU: `grep -A3 "MPI RANK" <name>.out` lists one device per rank.
- An old script with plain `import pyquick` runs on the CPU, not the GPU. `gpu_enabled()`
  shows which engine a script got.
- Known QUICK issues: persistent jobs leak a little device memory per step; on the multi-GPU
  engine, a gradient of a molecule with only s functions (H2/STO-3G) crashes, as it does in
  `quick.cuda.MPI`.

## Verify

```bash
python3 -c "
import pyquick
r = pyquick.PyQuick(method='HF', basis='STO-3G').get_energy(
    'O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24', name='verify')
print(r.total_energy)
"
```

The test suites (in `src/pyquick/test/`) pick the mode with `PYQUICK_MODE` (`serial` by default,
`mpi`, `gpu` or `gpu_mpi`):

```bash
python3 -m pytest test_pyquick_modes.py -q            # which engines exist, how modes are chosen
python3 -m pytest test_pyquick_api.py test_pyquick_xyz.py -q
PYQUICK_MODE=gpu python3 -m pytest test_pyquick_api.py -q
PYQUICK_MODE=mpi mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_api.py test_pyquick_mpi.py -q
PYQUICK_MODE=mpi python3 -m pytest test_pyquick_mpi_launch.py -q    # starts mpiexec itself
PYQUICK_MODE=gpu_mpi mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_gpu_mpi.py -q
```

`test_pyquick_modes.py` and `test_pyquick_mpi_launch.py` read `PYQUICK_MPIEXEC` (for example
`"mpiexec --oversubscribe"` when ranks > cores).

`runtest` runs the water energy test `test/pyq_ene_H2O_rhf_sto3g.py` in the matching mode:
`--serial` in serial, `--mpi` in `mpi` under `DO_PARALLEL`, `--cuda`/`--hip` in `gpu`, and
`--cudampi`/`--hipmpi` in `gpu_mpi` under `DO_PARALLEL`. It is skipped when that engine was
not built.

## Common pitfalls

- **"Python dependency not found" in meson** — your Python has no dev headers (see Prerequisites).
- **`meson`/`ninja` "not found" even though installed** — `src/pyquick/CMakeLists.txt` finds them
  via `PATH`, not relative to `-DPython3_EXECUTABLE`. Make sure that environment's `bin/` is on
  `PATH` at **both** configure and build time.
- **GPU build "succeeds" but the engine never touches the device** — f2py's meson backend has
  historically dropped `--f90flags` on some versions, silently compiling the `#if defined(GPU)`
  guards out. Verify: `nm -D <install>/lib/pyquick/_engines/gpu/_pyquick*.so | grep gpu_` should
  show undefined `gpu_*_` symbols. If it shows none, the flags were dropped.
- **An install from before the engine layout** still has `lib/pyquick/_pyquick*.so`. Nothing
  imports it any more, but delete it to avoid confusion.
- **One Python version at a time** — an engine built for a different Python version is not
  found (`not available`); rebuild after switching Python.
- **macOS: `runtest` skips the Python test** even though pyquick works in your shell. macOS
  removes `DYLD_LIBRARY_PATH` from programs started through `/bin/bash`, so the engines cannot
  find `libquick` there. Run the pytest suites directly instead.
