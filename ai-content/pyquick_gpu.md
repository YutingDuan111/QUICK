# pyquick on GPU (delorean)

Status: **done and verified on real hardware** — both the one-shot API (`get_energy`/`get_grad`/
`geo_opt`) and the persistent `Job` API (`job_open`/`job_step`/`job_close`) run on the GPU. AMD/HIP
is unstarted (same source, different build recipe — see "Not done yet").

Read together with `pyquick_progress_summary.md` (what pyquick is, its Python API).

## The core fact: GPU is compile-time, not runtime

QUICK's GPU code is behind `#if defined(GPU)` (see `src/main.f90:101,152,193,391`). It only exists
in a binary/extension compiled with `-DCUDA=TRUE` (or `-DHIP=TRUE`), which builds a separate
library `libquick_cuda` (`src/CMakeLists.txt:181,197`). **There is no `device=` argument, no
environment variable pyquick reads, nothing to toggle in code.** What decides whether a run uses
the GPU is simply which compiled `_pyquick*.so` gets imported, which depends on which install's
`PYTHONPATH` is active in your shell. Activate the GPU install and every run goes to the device;
activate a CPU install instead and the identical script runs on CPU — same test file, same code,
zero branching (confirmed: `test_pyquick_api.py` has no GPU/CPU logic in it at all).

**CPU build verified working too** (2026-08-19), as a regression check on the GPU work above —
the GPU code is all behind `#if defined(GPU)`, which should compile out cleanly for a CPU-only
configure, but that had never actually been checked on this box (only the GPU configure had been
built here). `src/pyquick/test/verify_cpu_build.sh` does a fresh `-DCUDA` (unset/default-`FALSE`)
configure + build + install into `build-gnu-cpu`/`install-gnu-cpu`, checks the installed `.so` has
**zero** `gpu_*` symbol references (`nm -D`), then runs the full pytest suite — all three passed
with **no `CMakeLists.txt` changes needed**. Both builds pass 29/29 independently; running one
doesn't disturb the other (separate build/install directories).

## What's already built on delorean

| | Path |
|---|---|
| GPU build dir / install prefix | `build-gnu-cuda/` / `install-gnu-cuda/` |
| CPU build dir / install prefix | `build-gnu-cpu/` / `install-gnu-cpu/` |
| GPU-linked pyquick module | `install-gnu-cuda/lib/pyquick/_pyquick.cpython-311-x86_64-linux-gnu.so` |
| CPU-linked pyquick module | `install-gnu-cpu/lib/pyquick/_pyquick.cpython-311-x86_64-linux-gnu.so` |
| Python interpreter (both builds) | `/server-home1/yud042/miniforge3/envs/pyquick-gpu/bin/python3.11` (3.11.15, numpy 2.4.6, meson, ninja, pytest) |

GPU built with `-DCOMPILER=GNU -DCUDA=TRUE -DQUICK_USER_ARCH=turing -DPYTHON=TRUE`; CPU built with
the same minus `-DCUDA`. No CUDA module load is needed at *run* time for the GPU build —
`libquick_cuda.so`'s RPATH already bakes in `/usr/local/cuda-13.1.1/lib64` — only when
(re)building.

## Running it

Three exports must be live in **whatever shell or script actually calls `python3.11`**:
`quick.rc` sets `QUICK_BASIS`/`LD_LIBRARY_PATH`; the `PATH` export is what makes `python3.11` (and
`meson`/`ninja`, needed only when rebuilding) resolve at all; `PYTHONPATH` is what selects the
GPU-linked `.so` over any CPU one. **Keep these in the same script/block as the code that uses
them** — splitting them (e.g. pasting only the last line of a multi-line example) fails with
`python3.11: command not found`, often silently if there's no `set -e` (a leftover `.out` from an
earlier run can then make a `grep` check look like it passed).

```bash
#!/bin/bash
set -e
source /server-home1/yud042/Documents/Github/QUICK/install-gnu-cuda/quick.rc
export PATH=/server-home1/yud042/miniforge3/envs/pyquick-gpu/bin:$PATH
export PYTHONPATH=/server-home1/yud042/Documents/Github/QUICK/install-gnu-cuda/lib:$PYTHONPATH

python3.11 -c "
import pyquick
calc = pyquick.Calculation(method='HF', basis='STO-3G', log=True)
r = calc.get_energy('O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24', name='my_run')
print(r.total_energy)
"
grep -i 'gpu\|device\|cuda' my_run.out   # confirms the device block is there
```

This exact script exists and works: `src/pyquick/test/test_GPU.sh` — run with `./test_GPU.sh`
from anywhere, doesn't depend on cwd or an already-active shell. Two more ready-to-run scripts:
`src/pyquick/test/verify_persistent_job_gpu.sh` (persistent-job GPU check) and
`src/pyquick/test/verify_cpu_build.sh` (fresh CPU-only build + test, see "CPU build" note above).

**API reminder** (differs from some older examples floating around): the constructor is
`Calculation(method, basis, properties=(), charge=0, mult=1, keywords=None, log=True)` — `name`
is **not** a constructor argument, it belongs to `get_energy(geometry, name=None)` (and likewise
`get_grad`/`geo_opt`). Note also: `job.run()` on a persistent `Job` has no `name=` parameter at
all — its log always uses whatever `output_stem` was last set to (Fortran module state that
persists across calls); if you called a *named* one-shot `get_energy(..., name=...)` earlier in
the same process, the persistent job's log silently lands in *that* file instead of the default
`pyquick_job.out`. Open the persistent job first if you want its own clean log.

### Confirming a run actually used the GPU

Every run prints a `GPU INFORMATION` block to the `.out` file (only if `log=True`, the default):

```
|------------ GPU INFORMATION ---------------
| CUDA ENABLED DEVICE         :        1
| CUDA DEVICE IN USE          :        0
| CUDA DEVICE NAME            : NVIDIA GeForce RTX 2080
...
```

If that block is missing, you imported a CPU-only `pyquick` — check `PYTHONPATH`. If you only see
the header line and none of the detail lines, that's a `grep` pattern problem, not a build
problem: only the header literally contains the substring `gpu`; the detail lines say
`CUDA`/`DEVICE` instead. Use `grep -i 'gpu\|device\|cuda'`, not `grep -i gpu`.

### Running the test suite

Run the whole block together, including the `cd` — the test path is relative to the repo root, so
running from inside `src/pyquick/test/` gives "file or directory not found" for that same path.

```bash
source /server-home1/yud042/Documents/Github/QUICK/install-gnu-cuda/quick.rc
export PATH=/server-home1/yud042/miniforge3/envs/pyquick-gpu/bin:$PATH
export PYTHONPATH=/server-home1/yud042/Documents/Github/QUICK/install-gnu-cuda/lib:$PYTHONPATH
cd /server-home1/yud042/Documents/Github/QUICK
python3.11 -m pytest src/pyquick/test/test_pyquick_api.py -q
```

No extra syntax, no marker, no `--gpu` flag — same command as CPU. **29/29 pass** against the GPU
build.

### Persistent jobs on GPU (`calc.new_job(...)`)

```python
job = calc.new_job(geometry)          # opens once: device context + persistent scratch
e1 = job.run(geometry).total_energy
e2 = job.run(moved_geometry).total_energy      # reuses the density from step 1 as its guess
r3 = job.run(another_geometry, gradient=True)
job.delete()                          # or use `job` as a context manager
```

Verified: a 4-step run alternating two geometries (one step with `gradient=True`) matches the
one-shot reference to ~1e-13; a one-shot `geo_opt()` interleaved with persistent-job use also
verified; a second persistent job opened after the first closed also verified.

**Known limitation** (not a correctness issue — see "Known bugs" below for the root cause): each
`job.run()` step leaks a basis-sized chunk of device memory, reclaimed only when the job closes.
Fine for short-to-moderate step counts; avoid very long-running persistent GPU jobs until the
underlying `gpu_cleanup()` bug is fixed.

## How this was built

### Source (`src/pyquick/pyquick.f90`)

GPU-guarded blocks (`#if defined(GPU) ... #endif`) in:
- **`job_run`** (one-shot path) — 4 blocks copied from `main.f90`: `use allmod` (exposes `gpu_*` +
  basis arrays, subroutine-scoped so each subroutine needs its own); device init
  (`gpu_new`/`gpu_init_device`/`gpu_write_info`) before `read_Job_and_Atom`; after `getMol`,
  `gpu_allocate_scratch` + `upload(quick_method)` + `gpu_setup`/`gpu_upload_xyz`/
  `gpu_upload_atom_and_chg`, then after `getEriPrecomputables`/`schwarzoff`,
  `gpu_upload_basis`/`gpu_upload_cutoff_matrix`/`gpu_upload_oei`; after `harvest_results`,
  `gpu_deallocate_scratch` + `gpu_delete`.
- **`job_open`/`job_step`/`job_close`/`job_destroy`** (persistent path) — modeled on
  `src/modules/quick_api_module.f90` (the real AMBER Fortran API), not guessed: `job_open` does
  the one-time device/context setup only (`gpu_new`/`gpu_allocate_scratch`/`upload(quick_method)`
  — **not** basis upload); `job_step` re-runs the *entire* upload sequence every call, because
  `gpu_upload_basis`/`gpu_upload_oei` also compute geometry-derived quantities (interatomic
  distances, Gaussian product centers) that go stale the instant coordinates change — uploading
  basis once in `job_open` and only `xyz` per step (an earlier, wrong assumption) would have
  produced silently-incorrect forces/energies from step 2 onward; `job_close`/`job_destroy` share
  a `job_gpu_teardown()` helper (see "Known bugs" for why it's gated on `job_open_flag`, not
  `job_active`).

### Build (`src/pyquick/CMakeLists.txt`, `quick-cmake/QUICKCudaConfig.cmake`)

1. Preprocessing is unconditional: `--f90flags="-cpp -I<repo>/src/util"` on every build (CPU
   included), so the `#if defined(GPU)` guards are evaluated — and compile out — on CPU too.
2. `if(CUDA)` appends `-DGPU -D${QUICK_GPU_PLATFORM}`, switches module dir to
   `amber-modules/quick/${QUICK_GPU_TARGET_NAME}` (`cuda` or `hip`), links
   `-lquick_${QUICK_GPU_TARGET_NAME}` instead of `-lquick`.
3. CUDA-only extra link args: `-L<dir> -lcudart -lcublas -lcusolver`, derived from FindCUDA's
   `CUDA_CUDART_LIBRARY`/`CUDA_cublas_LIBRARY`/`CUDA_cusolver_LIBRARY` (split into `-L`/`-l`
   because f2py's meson backend wants that form; named explicitly because f2py links with
   gfortran, not the CUDA toolchain, so these aren't transitively picked up otherwise).
4. HIP branch reuses the same target-name logic, skips the CUDA math libs (rocblas/rocsolver come
   in transitively via `libquick_hip`) — untested, no box with an AMD GPU yet.

### Python environment — resolved the hard way, do not repeat

Two separate failures, in order, building with the system Python:

1. **System python3.9 + numpy 1.23.5**: f2py's `--backend meson` flag didn't exist before numpy
   1.26.0, and `CMakeLists.txt` hardcodes it. Fix: numpy >= 1.26.
2. **A venv from the system python still failed**: meson died with `Run-time dependency python
   found: NO` → `Python dependency not found`. Cause: **no Python development headers** on the box
   (`python3-devel` not installed, no root to install it) — a venv inherits headers from its base
   interpreter, so it can't fix this.

Resolution: use an interpreter that ships its own headers + pkg-config. No `module avail python`
entry exists on delorean; `/server-home1/yud042/miniforge3` was present but an interrupted install
(`micromamba` binary + `pkgs/` cache, no `bin/python3`, no `envs/`). Finished it directly:

```bash
export MAMBA_ROOT_PREFIX=/server-home1/yud042/miniforge3
/server-home1/yud042/miniforge3/micromamba create -y -n pyquick-gpu -c conda-forge \
    python=3.11 "numpy>=1.26" meson ninja pkg-config pytest
```

Two things must both be true at configure **and** build time: `-DPython3_EXECUTABLE=.../envs/
pyquick-gpu/bin/python3.11`, **and** the env's `bin/` prepended to `PATH` —
`src/pyquick/CMakeLists.txt` finds `meson`/`ninja` via `find_program()` on `PATH`, not relative to
the interpreter, so pointing at the interpreter alone is not enough.

### Build commands actually used (two passes — don't skip the split)

Pass 1 with `-DPYTHON=FALSE` proves the CUDA toolchain works and `quick.cuda` runs on the device
*before* introducing f2py/Python into the picture — mixing the two makes a CUDA toolchain failure
look like an f2py failure.

```bash
# Pass 1: CUDA-only, no Python.
module load cuda/13.1.1
mkdir -p build-gnu-cuda && cd build-gnu-cuda
cmake .. -DCOMPILER=GNU -DCUDA=TRUE -DQUICK_USER_ARCH=turing -DPYTHON=FALSE -DENABLEF=FALSE \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install-gnu-cuda
cmake --build . --parallel $(nproc) && cmake --install .
source ../install-gnu-cuda/quick.rc
quick.cuda ../test/ene_H2O_rhf_sto3g.in && grep -i 'gpu\|device\|cuda' ../test/ene_H2O_rhf_sto3g.out

# Pass 2: turn on Python, same build dir.
export PATH=/server-home1/yud042/miniforge3/envs/pyquick-gpu/bin:$PATH
cmake .. -DPYTHON=TRUE -DPython3_EXECUTABLE=/server-home1/yud042/miniforge3/envs/pyquick-gpu/bin/python3.11
cmake --build . --parallel $(nproc) && cmake --install .
# configure should print: "pyquick: binding against GPU libquick (quick_cuda)"
```

**Verified**: `QUICK_USER_ARCH=turing` matches the box's RTX 2080 (SM 7.5); `module load
cuda/13.1.1` matches the driver's max-supported CUDA exactly; gcc 11.5 confirmed compatible by
`QUICKCudaConfig.cmake`'s own version check. `quick.cuda` on `test/ene_H2O_rhf_sto3g.in` produced
a `GPU INFORMATION` block and `TOTAL ENERGY = -74.947863811`, an **exact** match to the CPU
reference. `nm -D` on the installed `.so` showed undefined references to every `gpu_*_` symbol,
proving `-DGPU` reached gfortran (the guarded blocks compiled in, not out). pyquick's own
`get_energy` on the same input matched to ~1e-10. Note: the plain `quick.cuda` binary's `.out`
lands next to the input file; pyquick's `get_energy(..., name=...)` writes `<name>.out` to the
*cwd* instead.

## Known bugs and limitations

### 1. `gpu_cleanup()` segfaults — pre-existing, unrelated to pyquick, ROOT CAUSE FOUND, fix not applied yet (deliberate — see below)

`src/gpu/cuda/gpu.cu`'s `gpu_cleanup()` is supposed to free a job step's geometry-dependent
`gpu_basis`/`gpu_cutoff` device arrays between steps (`quick_api_module.f90`'s `run_quick` calls
it after every step). It segfaults. Confirmed pre-existing — not something this work
introduced — by adding the *identical* call to the already-proven one-shot `job_run` teardown:
same segfault. Never hit before because nothing in pyquick's GPU path had ever called it.

**Root cause (confirmed 2026-08-19 with a real debug backtrace)**: `gpu_upload_cutoff()`
(`gpu.cu:840`, called from `quick_scf_operator_module.f90`/`quick_uscf_operator_module.f90` —
**once per SCF iteration**, not once per job step) does
`gpu->gpu_cutoff->cutMatrix = new gpu_buffer_type<QUICKDouble>(...)` unconditionally, every call,
never freeing the previous iteration's buffer first. A single `get_energy()` call runs SCF to
convergence (typically 5-15 iterations), so this leaks that many orphaned buffers per calculation.
Confirmed by building a debug (`-g`) copy of `libquick_cuda.so` (compiled gpu.cu separately with
`-O0 -g -G`, relinked into a standalone `.so`, temporarily swapped into
`install-gnu-cuda/lib/libquick_cuda.so` for one test, then restored) and reproducing under gdb:
crash is `gpu_cleanup_() at gpu.cu:2913`, `SAFE_DELETE(gpu->gpu_cutoff->cutMatrix)`, with `cutMatrix`
holding a live-but-invalid pointer at that point — consistent with heap corruption from the
repeated leak-without-free pattern. Corroborating evidence: another cleanup routine elsewhere in
the same file (`gpu_addint_`, older/unused path) frees this exact same group of four `gpu_cutoff`
buffers together, confirming that's the codebase's own intended pattern — it's just missing at
the one call site that actually matters.

**Proposed fix (not applied — user decision, see "Decision" below)**: one line in
`gpu_upload_cutoff_` (`gpu.cu:855`), matching the `SAFE_DELETE` macro already used everywhere else
in this file:
```c
SAFE_DELETE(gpu->gpu_cutoff->cutMatrix);   // free the previous SCF iteration's buffer first
gpu->gpu_cutoff->cutMatrix = new gpu_buffer_type<QUICKDouble>(cutMatrix, gpu->nshell, gpu->nshell);
```

**Workaround shipped** (independent of the above — this is what's actually running today):
`job_step` does not call `gpu_cleanup()` at all. Each step's `gpu_setup`/`gpu_upload_basis`
`new`-allocate fresh device buffers without freeing the previous step's, reclaimed only at
`job_close`/`job_destroy` via `gpu_delete`'s `cudaDeviceReset()`.

**Practical impact — why this doesn't block current usage**:
- **Plain `quick` binary and pyquick's one-shot calls (`get_energy`/`get_grad`/`geo_opt`):
  unaffected in practice.** `gpu_cleanup()` is never called on this path — `job_run` goes straight
  to `gpu_delete()`/`cudaDeviceReset()`, which wipes *all* device memory regardless of what leaked
  during that one calculation. The per-SCF-iteration leak is real but reclaimed the instant the
  calculation ends; it doesn't accumulate across repeated one-shot calls.
- **Persistent `Job`: the one path where this matters, because the GPU context stays alive across
  many calls.** Two leaks stack per step: the workaround's basis-array leak (once per `job_step`
  call) plus this `cutMatrix` leak (once per SCF iteration *inside* each step, so several times
  more per step than previously documented here). Both scale roughly with system size squared
  (basis-function-squared / shell-squared arrays), so the tiny water/STO-3G test system leaks only
  ~10-20 KB/step (thousands of steps needed to matter), but a realistic production system could
  plausibly hit the 8 GB device limit within hundreds of steps or fewer. The failure mode is a
  hard `cudaMalloc` out-of-memory error partway through a long run, **not** a slowdown and **not**
  incorrect results — per-step compute speed and numerical correctness are both unaffected;
  everything verified so far (down to ~1e-13) is entirely trustworthy right up until the point
  memory runs out.

**Decision (2026-08-19): leaving this unfixed for now.** Acceptable for one-shot usage and for
persistent jobs with a bounded, modest step count on small/medium systems. Revisit before relying
on persistent `Job` for long trajectories or large systems — that's precisely the scenario this
leak targets and precisely the scenario the feature exists for. Fix is scoped and ready to apply
(one line, above) whenever that becomes the priority.

### 2. Double-free between `job_run` and `job_close`/`job_destroy` — fixed

`job_run` (one-shot) always left `job_active = .true.` at the end, even though it already tore
down its own GPU context (`gpu_deallocate_scratch` + `gpu_delete`); `gpu_delete_()`'s C++ never
nulls the global `gpu` pointer afterward, so it was left dangling. The persistent-job GPU teardown
in `job_close`/`job_destroy` was originally keyed off that same `job_active` flag, so it fired
*again* after any one-shot call (e.g. `geo_opt()`) went out of scope and hit `PyQuick.__del__` →
`job_destroy()` — a double-free on the dangling context. Caught by pytest: a test using `geo_opt`
reliably crashed the whole suite with glibc's "double free or corruption (!prev)".

**Fixed**: the GPU teardown is now gated on `job_open_flag` (only set by `job_open`, never touched
by `job_run`) instead of `job_active` — correctly distinguishes "a persistent job's context is
still open" from "the engine merely ran a one-shot job at some point." `job_destroy` was also
fixed to reset `job_open_flag = .false.` after tearing down (it previously only reset
`job_active`), matching `job_close` — otherwise a later `job_open()` on the same engine would
wrongly refuse with "a job is already open."

### 3. `pyquick.f90` build-caching gotcha

After editing `pyquick.f90`, `cmake --build . --target pyquick` can report `[100%] Built target
pyquick` and genuinely re-run f2py (visible in the log: "Reading file .../pyquick.f90") **without
the resulting `.so`'s actual compiled behavior changing** — confirmed by disassembling a changed
subroutine in the installed `.so` and finding the old code path, twice, after two separate
"successful" rebuilds. `touch`-ing the source first didn't help. What reliably worked: `rm -f
build-gnu-cuda/src/pyquick/_pyquick.cpython-311-x86_64-linux-gnu.so` **before** `cmake --build`.
Root cause not fully chased down. **Always verify a `pyquick.f90` rebuild actually changed
behavior** — e.g. `objdump -d --demangle .../_pyquick*.so | awk '/<__pyquick_MOD_<sub>>:/{f=1}
f{print} f&&/^$/{exit}'` and check for the expected instructions — before trusting a "successful"
build; a green build here is not proof.

### 4. f2py `--f90flags` risk — checked, did not occur

f2py's meson backend has historically dropped `--f90flags`. If it did, `-DGPU` would never reach
gfortran, the guarded blocks would compile out, and you'd get a **CPU-only `.so` that builds and
links cleanly and looks like it works**. Checked via `nm -D` on the installed `.so` for undefined
`gpu_*_` symbols (see "Build commands" above) rather than inspecting f2py's generated
`meson.build`, since f2py cleans up its temp build dir after producing the `.so`. Confirmed fine
on this box/toolchain; re-check if reproducing on a different CUDA/gfortran/numpy combination.

## Not done yet

- **AMD/HIP**: same source, `-DHIP=TRUE` build recipe (module dir `amber-modules/quick/hip`,
  links `libquick_hip`, platform macro still `GPU` plus `HIP`) — untested, no AMD GPU box
  available yet.
- **`gpu_cleanup()` fix** (see "Known bugs" #1 — root cause found, one-line fix scoped, deliberately
  not applied) — would remove the per-step/per-SCF-iteration memory leak in persistent GPU jobs.

## Workflow notes for this box

`delorean` — standalone RHEL 9 workstation, NVIDIA GPU (RTX 2080, Turing), no job scheduler (run
things directly, no `srun`). Home is `/server-home1/yud042`. Repo at `~/QUICK`, branch
`API-dev-GPU`. GCC/gfortran 11.5.0.

- **Long builds run in tmux, piped to a log**, never as an agent tool call: `tmux new -s build;
  cmake --build . 2>&1 | tee build.log`. A dropped SSH session then costs nothing.
- VS Code Remote-SSH for editing; Claude Code runs as the terminal TUI on delorean (the VS Code
  sidebar extension never activated and was abandoned).
