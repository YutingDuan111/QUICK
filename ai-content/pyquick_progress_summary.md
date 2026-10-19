# pyquick — progress summary (handoff)

State as of 2026-07-17. Branch **`API-dev-GPU`**. CPU build/tests are on a macOS + Python 3.9
(Homebrew) setup; GPU work is intended for an SSH box with NVIDIA + AMD GPUs.

See also: `ai-content/pyquick_development_steps.md` (full roadmap), `ai-content/quick_api_parameter_spec.md`
(energy model / data formats), and the memory files `pyquick-scope-overrides.md` /
`pyquick-test-first-workflow.md`.

## What pyquick is

A Python interface to QUICK via f2py (`src/pyquick/pyquick.f90` = Fortran binding wrapping
`libquick`; `src/pyquick/__init__.py` = the `Calculation`/`Result`/`Job` classes over a
low-level `PyQuick` engine). Goal: automate molecular dataset generation for ML. In-memory,
no output-file parsing.

## Done and working (47 pytest tests pass: `src/pyquick/test/test_pyquick_api.py`)

- **One-shot API**: `Calculation(method, basis, properties=(), charge, mult, keywords, log)` →
  `get_energy(geom, name=None)`, `get_grad(geom, name=None)`, `geo_opt(geom, name=None, max_cycles=None)`.
  Job type = the method you call. Each returns a `Result`.
- **Energy model** (corrected): `total = nuclear_repulsion + e_electronic (+ e_dispersion)`;
  `e_electronic = e_one_electron + e_two_electron + e_xc`. `e_two_electron` derived exactly.
  `e_external_charge` never in total. (`ECore` → `nuclear_repulsion`.)
- **Always on the Result**: energies, geometry (`atomic_numbers`/`coordinates`, from QUICK's
  parse), `mo_energies`, `density_matrix` — every SCF computes them for free.
- **Opt-in via `properties`**: `mulliken_charges`, `lowdin_charges`, `dipole` (all need the
  DIPOLE keyword). `AttributeError` (with reason) if accessed unrequested.
- **Gradients** (`get_grad`): SCF+gradient together; `Result.gradient` `(natom,3)` Hartree/Bohr.
  Validated by finite difference.
- **geo_opt**: **DL-Find is the default** optimizer; `keywords={'LOPT': None}` picks Cartesian.
  DL-Find crashes on <3 atoms → `geo_opt` raises `ValueError` pointing to LOPT (Python guard +
  Fortran safety-net). `Result.optimized_coordinates` (Å), `Result.converged`; `coordinates`
  stays the input. `max_cycles=None` = bare `OPTIMIZE` (run to convergence).
- **Dipole** (Phase 2.0): stored field `quick_qm_struct%dipole` (Debye), set in `src/subs/dipole.f90`.
- **Logging**: `log=True` (default) writes `<name>.out` (name from `get_energy(..., name=)`,
  else `pyquick_job.out`); `log=False` → platform null device (`NUL` Windows / `/dev/null`),
  no file. Replace-then-append across runs. Null device chosen at runtime by `null_device()`.
- **Persistent `Job`** (AMBER-API style): `calc.new_job(geom)` → `Job`; `job.run(geom, gradient=False)`
  steps successive geometries of the **same** molecule reusing the in-place density
  (`getMol` not re-called between steps); `job.delete()` or `with`. Fortran: `job_open`/
  `job_step`/`job_close` in pyquick.f90; harvest factored to `harvest_results(did_gradient)`.
  One active job (global singleton). Verified: stepped E == one-shot to ~1e-7.
- **No serialization in the API** — `Result` gives floats/NumPy; users save with `h5py`.

## Error handling (documented limitation, NOT fixed)

QUICK's engine calls `exit()` on error (`SAFE_CALL`/`CHECK_ERROR` → `quick_exit`), killing the
Python process/kernel — **not catchable** with `try/except`. Only pyquick's own binding-level
validation (geometry parse, setup order) raises catchable `RuntimeError`/`ValueError`.
Best practice: with `log=True`, the error lands in `<name>.out` on disk (survives the dead
kernel) — read it after restart. A stderr-echo stopgap was tried and reverted (useless in
Jupyter). `fail_quick` was removed (no reachable trigger). The real fix ("exit-free apiMode")
is scoped but not done. Full write-up: the user has a standalone error-handling doc.

## Notebooks (`src/pyquick/test/`)

- `usecase1_reaction_ene.ipynb` — C₂H₄+H₂→C₂H₆ reaction energy (Calculation/Result, h5py save).
- `usecase2_permentant_job.ipynb` — one-shot vs persistent Job workflows (spec + demo).
- `usecase3_basis_convergence.ipynb` — C₂H₆ energy vs basis (HF/BLYP/B3LYP × 3-21G→6-311G**).

Notebooks run with the Python 3.9 kernel (has rdkit/numpy); source `install-API/quick.rc` before
launching Jupyter so `PYTHONPATH`/`QUICK_BASIS`/`DYLD_LIBRARY_PATH` are set.

## Build / run reminders (CPU)

```bash
cd builddir && cmake .. -DCOMPILER=CLANG -DPYTHON=TRUE \
    -DPython3_EXECUTABLE=/opt/homebrew/bin/python3.9 -DCMAKE_INSTALL_PREFIX=../install-API
make -j && make install
source ../install-API/quick.rc
/opt/homebrew/bin/python3.9 -m pytest src/pyquick/test/test_pyquick_api.py -q   # 47 pass
```
- f2py needs `meson` + `ninja` for python3.9; the f2py command uses `--backend meson`.
- Only ever build/install for **one** Python (3.9); a stale cross-version `_pyquick.*.so`
  shadows it and causes failures.

## Git state

Branch `API-dev-GPU`. Committed: `geo_opt() enabled`, `permenant run` (pyquick.f90 + __init__.py).
**Uncommitted** on the Mac working tree: `test_pyquick_api.py`, `pyquick_ene_H2O_rhf_sto3g.py`,
the GPU block in `pyquick.f90` (see the GPU doc), `src/pyquick/README.md`, the three notebooks,
and `ai-content/`. `builddir/`, `install-API/`, `.DS_Store`, `.vscode/`, `mytest/` must NOT be
committed. No remote push yet — `origin git@github.com:YutingDuan111/QUICK.git`.

## Next steps

1. Commit + push the working tree; pull on the SSH box.
2. **GPU** — see `ai-content/pyquick_gpu.md`. The guarded GPU block is already in
   `pyquick.f90`; the build must be changed to preprocess + link the GPU library.
3. (Later) per-step GPU uploads in `job_step` for the persistent Job; ZPE/frequencies are broken
   in QUICK (`FREQ` segfaults) so out of scope; cutoff named-parameters are optional (the
   `keywords={}` passthrough already covers every cutoff keyword).
