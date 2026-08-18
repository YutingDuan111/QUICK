# pyquick GPU enablement — approach (for the SSH GPU session)

Goal: run pyquick calculations on the GPU, the same way the `quick` binary does. Target box has
**NVIDIA + AMD** GPUs. Do **CUDA (NVIDIA) first** (QUICK's most-tested path), then AMD via HIP is
the same source with a different build recipe. Single-GPU only — **no MPI / GPU-MPI** for now.

## The core fact: GPU is compile-time, not runtime

QUICK's GPU code is behind `#if defined(GPU)` (see `src/main.f90:101,152,193,391`). It only exists
in the binary if compiled with `-DCUDA=TRUE` (or `-DHIP=TRUE`), which builds a **separate library
`libquick_cuda`** (`src/CMakeLists.txt:181`, `197`: `target_compile_definitions(... PRIVATE GPU ...)`).
Neither Jupyter nor Python detects or enables a GPU. A CPU-built `_pyquick.so` never touches the
device no matter what hardware is present. So enabling GPU = **source (done) + build change (to do)**.

The macro is `GPU` (defined for BOTH cuda and hip builds), so the Fortran block below is
**backend-agnostic** — one source, two build recipes.

## What is already done (source)

The guarded GPU block is **already in `src/pyquick/pyquick.f90`** inside `job_run`, copied from
`main.f90`, in 4 places (all `#if defined(GPU) ... #endif`):

1. `use allmod` at the top of `job_run` (exposes `gpu_*` + basis arrays).
2. after the log opens, before `read_Job_and_Atom`: `gpu_new`, `gpu_init_device`, `gpu_write_info`.
3. after `getMol`: `gpu_allocate_scratch`, `upload(quick_method)`, `gpu_setup`, `gpu_upload_xyz`,
   `gpu_upload_atom_and_chg`; then after `getEriPrecomputables`/`schwarzoff`: `gpu_upload_basis`,
   `gpu_upload_cutoff_matrix`, `gpu_upload_oei`.
4. after `harvest_results`: `gpu_deallocate_scratch`, `gpu_delete`.

It is committed to the branch. It is inert unless the build preprocesses the file.

## The build change — WRITTEN, needs GPU testing

`src/pyquick/CMakeLists.txt` has been updated (sections 4 + 5). It now selects the libquick flavour
from the `CUDA` option:

1. **Preprocessing is unconditional**: `--f90flags="-cpp -I<repo>/src/util"` on every build, so the
   guards are evaluated (and compile out) on CPU too. `-I src/util` is required once `-cpp` is on,
   because `#include "util.fh"` is then resolved by cpp rather than by gfortran's own handling.
2. **`if(CUDA)`** additionally appends `-DGPU -D${QUICK_GPU_PLATFORM}`, switches the module dir to
   `amber-modules/quick/${QUICK_GPU_TARGET_NAME}` (`cuda` or `hip`), and links
   `-lquick_${QUICK_GPU_TARGET_NAME}` instead of `-lquick`.
3. **CUDA-only extra link args**: `-L<dir> -lcudart -lcublas -lcusolver`, derived from FindCUDA's
   `CUDA_CUDART_LIBRARY` / `CUDA_cublas_LIBRARY` / `CUDA_cusolver_LIBRARY` full paths (split into
   `-L`/`-l` because f2py's meson backend wants that form). `libquick_cuda` is a *shared* library
   (`src/CMakeLists.txt:129` + `copy_target`), so these are normally transitive — they are named
   explicitly only because f2py links with gfortran, not the CUDA toolchain.
4. The HIP branch reuses the same target-name logic and skips the CUDA math libs; rocblas/rocsolver
   come in transitively via `libquick_hip`. Untested.

**Verified so far**: the CPU path only — reconfigured and rebuilt on the Mac, 47/47 pytest pass, so
the `-cpp` change does not regress CPU. Nothing in the GPU branch has ever been configured.

### If the f2py link leaves device symbols unresolved

QUICK's CUDA build may use separable/relocatable device code, which needs an `nvcc -dlink` step
that `ld`/`gfortran` (f2py's linker) can't do. Fallback: let **f2py only generate the wrapper +
compile `pyquick.f90` to objects**, then do the **final link in CMake with the CUDA toolchain**,
reusing the exact library set of the `test-api.cuda` target. Try the plain f2py link first; fall
back only if you hit unresolved `__cuda*` / device symbols.

## Build + verify on the SSH box

Use a **separate build dir** (`build-gpu/`) — do not reuse the CPU `builddir/`.

Do it in two passes. Pass 1 with `-DPYTHON=FALSE` proves the CUDA toolchain works on the box and
`quick.cuda` runs on the device; only then turn on `-DPYTHON=TRUE`. Mixing the two makes a CUDA
toolchain failure look like an f2py failure.

```bash
# 1. Build QUICK itself with GPU and confirm the plain binary uses the device first.
mkdir -p build-gpu && cd build-gpu
cmake .. -DCOMPILER=GNU -DCUDA=TRUE -DQUICK_USER_ARCH=<arch e.g. volta/ampere> \
    -DPYTHON=TRUE -DPython3_EXECUTABLE=<python> -DCMAKE_INSTALL_PREFIX=../install-GPU
make -j && make install
source ../install-GPU/quick.rc
# configure should print: "pyquick: binding against GPU libquick (quick_cuda)"

# 2. Run a pyquick energy and confirm it ran on the GPU.
python -c "import pyquick; print(pyquick.Calculation(method='HF', basis='STO-3G', \
    log=True, name='g').get_energy('O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24').total_energy)"
grep -i 'gpu\|device\|CUDA' g.out   # gpu_write_info output confirms the device was used
```

Install these for the build Python first (like the CPU build): `meson`, `ninja` (+ numpy).

## First compile errors to expect

- **Module interface / argument mismatches** in the `use allmod` block or the `gpu_upload_basis`
  argument list — the CUDA `.mod` files (absent on the Mac, present here) are the source of truth;
  the compiler pins them down. The arg lists were copied verbatim from `main.f90`, so they should
  match, but verify.
- **`gpu_write_info` / `upload(quick_method)`** need the right modules via `allmod`; if `allmod`
  conflicts with the module-level `use ... only:` imports, switch the block to `use`ing the
  specific gpu module instead.

## AMD (HIP) — after CUDA works

Same source block. Build recipe: `-DHIP=TRUE`, link `libquick_hip` + rocsolver/ROCm libs via
`hipcc`, module dir `amber-modules/quick/hip`, and the platform macro is still `GPU` (plus `HIP`).
No second implementation — just the analogous CMakeLists branch.

## Follow-ups (after job_run works on GPU)

- **Persistent `Job`**: `job_open`/`job_step` also need the GPU uploads (`gpu_upload_xyz` per step,
  `gpu_setup`/`gpu_upload_basis` once in `job_open`), mirroring `quick_api_module`'s per-step
  `gpu_upload_molspecs`. Do this only after the one-shot `job_run` GPU path is proven.
- Keep the CPU build working: the CMakeLists change must be conditional so a non-GPU configure
  still builds the serial CPU extension unchanged.
