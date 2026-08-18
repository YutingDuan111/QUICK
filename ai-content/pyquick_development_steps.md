# pyquick — Development Steps

A v1-first roadmap for finishing the `pyquick` Python interface to QUICK.
Grounded in the current source (`src/pyquick/pyquick.f90`, `src/pyquick/__init__.py`,
`src/pyquick/CMakeLists.txt`), the design doc
(`ai-content/Python Interface for QUICK Design Document.docx.pdf`), and the property
spec (`ai-content/quick_api_parameter_spec.md`).

## Guiding decisions (override the design doc where they conflict)

- **User-facing API = `Calculation` + `Result` (two classes).** Adopt the design's
  `Calculation` (holds settings, `run()` → `Result`) and `Result` (properties as
  attributes; `AttributeError` if a property wasn't requested/computed). Do **not** adopt
  `Molecule` — geometry is still passed via the existing string/`read_geom` path.
- **Keep the current pyquick *conventions* underneath.** Preserve the existing binding-layer
  style and semantics: `check()`-style `had_error`/`error_message` error handling, in-memory
  property transfer (no file parsing), per-run result snapshotting. The current `PyQuick`
  class becomes the low-level engine that `Calculation`/`Result` wrap — its conventions win
  on any conflict with the design doc.
- **`BatchRunner` is out of scope.** No batch orchestration, no `from_yaml`, no
  `n_workers` multiprocessing.
- **No serialization in the API.** `Result` exposes plain floats and NumPy arrays; the API
  writes no files. Users serialize datasets themselves (e.g. a few lines of `h5py`), which
  keeps schema/compression/resume under their control. (Decision 2026-07-06 — was previously
  "per-molecule HDF5 only"; `Result.to_hdf5` removed.) No JSON/ASE/combined-dataset output either.
- **QUICK's log is written by default; opt out with `log=False`** (decision 2026-07-16,
  reversing the 07-06 "no file I/O at all"). `Calculation(..., log=True)` is the default and
  writes QUICK's diagnostic output to `<stem>.out`, matching the pre-API behaviour.
  `log=False` sends it to the platform null device instead, so the run leaves nothing on disk.
  - **`name=` drives the log filename**: `get_energy(geom, name='C2H4')` → `C2H4.out` (and
    `name` still lands in `Result.metadata`). No name → `pyquick_job.out`.
  - **Re-run mode**: first run replaces the file, later runs on the same engine append;
    `quick_open` backs up a pre-existing file to `<file>~`.
  - The stem is set on **every** run, never only when a name is given — `output_stem` is Fortran
    module state that otherwise persists and silently redirects a later un-named run into the
    previous run's file. (Caught by `test_log_uses_default_stem_without_name`.)
  - The null-device path is chosen at **runtime** by `null_device()` — `NUL` on Windows
    (detected via `OS=Windows_NT` / `%WINDIR%`), `/dev/null` on Unix/macOS — needing no
    preprocessor flags or compiler-specific macros (Windows is a supported target).
  - Auxiliary QUICK files (`.dat`, `.molden`, `.vdw`) still appear only if their keyword
    (`CHK_WRITE`, `MOLDEN`, `ESP_CHARGE`) is explicitly set.
  - Tests that run SCF pass `log=False` so the suite leaves no `.out` files in the repo.
- **Errors: engine failures are NOT catchable today** (2026-07-10, revised 07-17).
  Catchable `RuntimeError`s come only from pyquick's own binding-level validation
  (`read_geom` parse errors, setup-order errors, keyword-line overflow) — raised *before*
  the engine runs. Every QUICK-engine failure calls `exit()` via `SAFE_CALL`/`CHECK_ERROR`
  *inside* the failing routine, so no error code ever returns to `job_run` (verified in
  source: `getMol` sets `ierr=18/19/20` then hits `CHECK_ERROR` immediately).
  > **`fail_quick` was removed** (2026-07-17): it composed QUICK's ierr→message into the
  > RuntimeError, but had **no reachable trigger** — dormant code. Kept: core
  > `get_exception_message(ierr,msg)` (used by `print_exception`; single source of truth)
  > and the working diagnostic lookup `pyquick._exception_message(code)` (+tests).
  >
  > **A stderr-echo stopgap was implemented and then reverted** (2026-07-16): useless in
  > Jupyter — ipykernel pipes fd 2 through a background thread and `exit()` outruns it, so
  > the message is lost from both cell and terminal. Do not retry.
  >
  > **Current best practice:** logging is on by default, and QUICK writes its error +
  > "Error Termination" to `<name>.out` **before** exiting — the log file survives the dead
  > kernel and is the only record of why (verified: bad basis → reason in `failed_job.out`).
  >
  > **The real fix is exit-free API mode** (`ierr` propagates instead of `exit()`). Not done.
  > Sizing: 61 `SAFE_CALL` + 26 `CHECK_ERROR` + 83 direct `call quick_exit` sites; `SAFE_CALL`
  > does not return from its enclosing routine, so each API-path site needs auditing. First
  > step: reachability audit of the routines `job_run` actually calls.
- **In-memory transfer, no file parsing** for properties (already the pattern in `pyquick.f90`).
  This includes **reusing QUICK's own routines** rather than re-implementing them in Python
  (e.g. geometry parsing / element lookup live in the Fortran layer — expose the parsed result,
  don't re-parse in Python).
- **v1 = bind what already exists** in QUICK; defer anything needing new Fortran-side
  instrumentation (`[DEV]` in the spec) to later phases.
- **During development, tests live under `src/pyquick/test/` and are run manually**
  (`pytest`, or `python3` directly) — **not** through the runtest harness or the install
  process. Every test for this API goes there for now. At PR time the smoke test will be
  moved to the standard top-level test directory and wired into runtest + install; that
  harness/install work is **deferred until then** (no install-rule change now).
- **Test-first workflow, every phase.** For each phase: (1) write/adjust the tests that encode
  the intended behavior, (2) **ask the user to check the tests**, (3) implement the code,
  (4) run the tests. Do not implement a phase's code before its tests are approved.

## Current state (baseline — already done)

- Fortran binding `src/pyquick/pyquick.f90`: `set_calc/set_basis/set_method/unset/clear/
  read_geom/run/destroy/set_output`; harvests in-memory `ETot`, `ECore`, `EEl`, `E1e`,
  `Exc`, `Edisp`, Mulliken, Löwdin, alpha MO energies, alpha density matrix.
- Python wrapper `src/pyquick/__init__.py`: `PyQuick` class with `_checked` error wrapping,
  result snapshotting per run, `copy()`, `input_string`.
- Build: f2py + CMake (`-DPYTHON=TRUE`), installs to `lib/pyquick`.
- Tests now in `src/pyquick/test/`: `pyquick_ene_H2O_rhf_sto3g.py` (smoke) and
  `test_pyquick_api.py` (pytest). Runtest/install references still point at the old
  top-level `test/` location and need updating (see §1.6).
- Removed: `src/pyquick/verify.f90` (was an unimplemented `compute_energy` stub).

---

## Phase 1 — v1: solidify and round out what's bindable today

Goal: a dependable single-molecule energy API returning every property QUICK already
computes, exposed through `Calculation` + `Result`, with HDF5 output and real tests.

### 1.1 Clean up the scaffold
- [ ] Remove `src/pyquick/verify.f90` (dead stub) or fold any useful comments into
      `pyquick.f90`. Confirm nothing in `CMakeLists.txt` references it.

### 1.2 Introduce the `Calculation` and `Result` classes
Wrap the existing `PyQuick` engine; do not rewrite the binding layer.
- [ ] **`Calculation`** in `src/pyquick/__init__.py`. Unified constructor:
      `Calculation(method=..., basis=..., properties=[...], charge=0, mult=1, keywords={...})`.
      - `method`: `'HF'`/`'UHF'` (explicit) **or** a DFT functional name (`'B3LYP'`, `'PBE0'`,
        …) which implies `DFT`/`UDFT`. Resolution: `HF`→`UHF` when `mult>1`; functional→`UDFT`
        when `mult>1`, else `DFT` (with the functional added as a keyword token).
      - `charge`/`mult` → `CHARGE=`/`MULT=` keyword tokens (verified parsed in apiMode at
        `quick_molspec_module.f90:424`).
      - `keywords`: extra passthrough tokens (`{'cutoff': '1e-9'}` → `CUTOFF=1e-9`; bare flags
        as `{'flag': None}`).
      - `properties` = **optional extras only** (default empty): `mulliken_charges`,
        `lowdin_charges` (both auto-add `DIPOLE`), `mo_energies`, `density_matrix`. The energy
        breakdown is always returned, so `energy` is **not** a property. Deferred-extra names
        (`dipole`, `esp_charges`, `beta_*`, `mp2`) raise a clear "planned for Phase N" error;
        anything else (including job-type words like `gradient`/`optimize`) is a plain
        "unknown property" error.
      - Validation reuses the `check()`-style errors surfaced by `_checked`.
- [ ] **Job type = the method called** (decision 2026-07-06): `get_energy(geometry, name=None)`
      (single point), `get_grad(...)` (energy + forces), `geo_opt(...)` (optimized geometry).
      Each maps settings → the `PyQuick` calls, snapshots into a **`Result`**, and returns it.
      Geometry is the existing string form (no `Molecule` class); one engine is reused across
      calls. In this phase only `get_energy` is implemented; `get_grad`/`geo_opt` raise
      `NotImplementedError("Phase 2")`.
- [ ] **`Result`**: owns the snapshot dict and the requested-property set + metadata.
      Energy attributes follow the corrected model (spec §5):
      `total_energy`, `nuclear_repulsion` (core–core; **renamed** from `ECore`/`e_core`),
      `e_electronic`, `e_one_electron` (renamed from `e_1e`), `e_two_electron`
      (**derived** `= e_electronic − e_one_electron − e_xc`; J+K combined, not split), `e_xc`.
      Conditional energies: `e_dispersion` (only when a `D*` keyword is requested; only then in
      `total_energy`) and `e_external_charge` (only when `EXTCHARGES` is set; **never** in
      `total_energy`). Other attributes: `mulliken`, `lowdin`, `mo_energies`, `density_matrix`.
      Raise `AttributeError` if a value is unavailable, and the message must state **why**
      and **how to enable it**:
      - not requested via `properties` (e.g. `mulliken`, `density_matrix`) → name the property
        to add to `Calculation.properties`;
      - driven by an input keyword (`e_dispersion` → a `D*` keyword; `e_external_charge` →
        `EXTCHARGES`) → say the keyword was not set and how to set it (via `keywords=`);
      - requested but not computed → say so.
- [ ] Energy model to preserve: `total_energy = nuclear_repulsion + e_electronic
      (+ e_dispersion)`; `e_electronic = e_one_electron + e_two_electron + e_xc`.
- [ ] Keep `PyQuick` available as the low-level engine (back-compat); `Calculation`/`Result`
      are the documented surface.

> **Dipole deferred** (decision 2026-06-26): the dipole vector is computed in
> `src/subs/dipole.f90` but never stored, so surfacing it needs a core-QUICK change.
> Moved to Phase 2 (lands with gradients). Phase 1 still sets `DIPOLE` internally when
> Mulliken/Löwdin are requested (that routine is what computes the charges).

### 1.3 Geometry access (for user-side serialization)
The API does not write files; users serialize `Result` data themselves. Give them the
geometry they need alongside the energies/arrays.
- [ ] **Reuse QUICK's parsed geometry**: bind the engine's already-parsed
      `geom_atnum`/`geom_coords` (Å) via `job_get_geometry`, and expose `Result.atomic_numbers`
      and `Result.coordinates`. **Do not** re-parse the geometry string or carry a Python-side
      periodic table.
- [ ] Density matrix stays an in-memory NumPy `float64` array of shape `(nbasis, nbasis)`
      from `quick_qm_struct%dense` (spec §4 density-matrix note).
- [ ] Document the "save it yourself" pattern (a few `h5py` lines; skip-if-exists for
      crash-safe batch resume) in the README and the demo notebook.

### 1.4 Tests for v1 (write first — see workflow)
All tests live in `src/pyquick/test/`.
- [ ] Add `pytest` unit tests covering: `Calculation` setup validation errors (missing
      method/basis/geom), method-token handling, `Result` attribute access including the
      `AttributeError` path, and re-running a `Calculation`.
- [ ] Energy-model tests: `total_energy ≈ nuclear_repulsion + e_electronic`;
      `e_electronic ≈ e_one_electron + e_two_electron + e_xc`; `e_xc == 0` for HF; components
      finite; `e_dispersion`/`e_external_charge` raise `AttributeError` when not requested/set.
- [ ] Property tests: Mulliken/Löwdin sum ≈ total charge; density-matrix square shape.
- [ ] Geometry test: `atomic_numbers`/`coordinates` come from QUICK's parse and round-trip
      the input geometry.
- [ ] Keep/extend the reference-energy smoke test
      (`src/pyquick/test/pyquick_ene_H2O_rhf_sto3g.py`) on `Calculation`/`Result`. Run it
      **manually** during development; do **not** wire it into runtest/install yet (deferred
      to PR — the smoke test then moves to the standard test dir).

### 1.5 Docs
- [ ] Update module/class docstrings; add a short `README`/usage snippet under `src/pyquick/`
      showing the `Calculation` → `get_energy`/`get_grad`/`geo_opt` → `Result` flow.

### 1.6 Realignment TODO (Phase 1 was implemented before these decisions)
The initial Phase 1 code predates the energy-model, geometry-reuse, and test-location
decisions above. Following the test-first workflow, adjust tests first, then:
- [x] Rename energy attributes/bindings: `e_core`→`nuclear_repulsion`, `e_1e`→`e_one_electron`,
      `e_disp`→`e_dispersion`; add `e_two_electron` (derived) and `e_external_charge`.
- [x] Add Fortran bindings: `job_e_charge` (`ECharge`) + availability flags
      `job_has_dispersion` (`quick_method%edisp`), `job_has_extcharge` (`quick_method%extcharges`).
- [x] Add a geometry binding (`job_get_geometry` → `geom_atnum`, `geom_coords`), expose
      `Result.atomic_numbers`/`coordinates`, and drop the Python `_parse_geometry` + `_PERIODIC_TABLE`.
- [x] Remove `Result.to_hdf5` and the numpy import; serialization is user-side now (2026-07-06).
- [x] Demo notebook `src/pyquick/test/usecase1.ipynb` rewritten for `Calculation`/`Result`
      (reaction energy, energy decomposition, charges, user-side h5py serialization, error demo);
      run it after rebuild to populate outputs.
- [x] Tests moved into `src/pyquick/test/` (run manually; harness/install wiring deferred to PR).
- [x] Split `Calculation.run` into `get_energy` / `get_grad` / `geo_opt` (job type = method,
      2026-07-06). `get_energy` works; `get_grad`/`geo_opt` raise `NotImplementedError("Phase 2")`.
      Dropped `energy` from `properties` (energy always returned; `properties` = extras only).

---

## Phase 2 — dipole, gradients and geometry optimization

Goal: cover the dipole moment plus `GRADIENT` and `OPTIMIZE` job types (spec §3).

### 2.0 Expose the dipole moment (deferred from Phase 1) — DONE 2026-07-09
- Spec status: computed in `src/subs/dipole.f90` but **never stored** — needed a small
  additive core-QUICK change.
- [x] Added a `dipole(3)` field to `quick_qm_struct` (`quick_calculated_module`) and store
      `xdip/ydip/zdip` there in `dipole.f90`, in **Debye** (matches the printed values and the
      design's `[0,0,1.85]` example).
- [x] Bind `job_dipole(3)` + `job_has_dipole` in `pyquick.f90`; added a `dipole` property on
      `Result`; added `dipole` to the supported `properties` vocabulary (needs `DIPOLE`).

### 2.1 Nuclear gradients — implement `Calculation.get_grad` — DONE 2026-07-09
- Spec status: `[API]` via `getQuickEnergyGradients` → `gradients(3,natoms)`.
- [x] `job_run(jobtype)` takes a job-type arg; for a gradient job it appends the `GRADIENT`
      keyword (so `quick_method%grad` is set before `getMol` allocates the array) and calls
      `cshell_gradient`/`oshell_gradient` instead of `getEnergy` — the gradient routine runs
      the SCF itself, so energy is not computed twice (mirrors `quick_api_module`).
- [x] Bind `job_get_gradients(g, n)` (flat `3*natom`, Hartree/Bohr); `Result.gradient` reshapes
      to `(natom, 3)`. Named `gradient` (raw dE/dR), not `forces`, to avoid a sign flip.
- [x] Implemented `Calculation.get_grad` (energy + gradient in one Result).
- [x] Test: central finite-difference of the energy reproduces the analytic gradient
      (`test_gradient_matches_finite_difference`) + translational-invariance check.

> **`get_grad` semantics** (decision 2026-07-09): stateless — always runs SCF+gradient together
> and returns both energy and gradient. No separate energy calc; no cross-call caching.

### 2.2 Geometry optimization — implement `Calculation.geo_opt` — DONE 2026-07-16
- Spec status: optimized coordinates `[FIELD]/[FILE]`.
- [x] `job_run(jobtype=2, ...)` appends `OPTIMIZE` (sets `quick_method%opt`+`%grad`) and calls
      **DL-Find** (`quick_method%usedlfind`, QUICK's default) or **`lopt`** when the user added
      the `LOPT` keyword. `lopt` is a **module procedure** (`use quick_optimizer_module`), not
      external; `dl_find` is external.
- [x] **Optimizer decision (2026-07-17, revised after mentor review):** DL-Find is the default,
      matching QUICK's CLI; users select Cartesian with `keywords={'LOPT': None}`. DL-Find
      crashes (BLAS error) on molecules with **< 3 atoms**, so `geo_opt` **guards** it:
      `Calculation.geo_opt` raises `ValueError` (pointing to `LOPT`) when a < 3-atom geometry is
      passed without `LOPT`, and `job_run` has a Fortran safety-net (`geom_natom < 3` +
      `usedlfind` → `fail`) so the low-level path can't segfault either. Atom count via
      `_count_atoms` (a line count, not a full parse). *(Supersedes the earlier
      "Cartesian-only, DL-Find not used" note.)*
- [x] `job_get_optimized_geometry` binds `quick_molspec%xyz` (which the optimizer updates
      **in place, in Bohr**) → Å at the boundary. `Result.optimized_coordinates`, `(natom,3)`.
      `Result.coordinates` still holds the **input** geometry.
- [x] **`Result.converged`**: both optimizers kept convergence in a local (`done` in `lopt`,
      `tconv` in `dlf_run`), so added `quick_qm_struct%opt_converged`, set in **both** (the
      `dlf_run` line is needed now that DL-Find is the default path).
- [x] `max_cycles=None` (default) emits **bare `OPTIMIZE`** = QUICK's own behaviour;
      `max_cycles=n` emits `OPTIMIZE=n`.
- [ ] (Optional, later) per-step trajectory — defer unless needed; lives in Molden today.

> **Corrections to earlier notes in this file** (verified empirically 2026-07-16, do not
> re-derive): `iopt` defaults to **0**, and bare `OPTIMIZE` leaves it 0 — but 0 means **no cap /
> run to convergence**, *not* zero steps. Verified: bare `OPTIMIZE` converges in 8 steps
> (DL-Find) and 10 (`LOPT`); `OPTIMIZE=2` stops early and warns. DL-Find's `maxcycle=100`
> default at `dl_find.f90:2044` only applies when `maxcycle < 0`, and the `!200` at
> `dlfind_main_driver.f90:199` is a **comment**, not a hardcoded default.
> Validated: water HF/STO-3G O–H 1.072 → 0.989 Å, E = −74.965901215 Ha (matches the CLI).

---

## Phase 3 — open-shell, MP2, and remaining stored fields

Goal: properties that exist as stored fields but aren't bound yet (spec §4–5).

- [ ] **Beta MO energies** (`Eb`, UHF/UDFT): `job_get_beta_mo_energies` + `beta_mo_energies`
      property on `Result`, gated on open-shell.
- [ ] **Beta density matrix** (`denseb`): bind + `beta_density_matrix` property on `Result`.
- [ ] **MP2 correlation energy** (`EMP2`): drive the MP2 path when `MP2` method set; bind
      `job_e_mp2` + `e_mp2` property on `Result`.
- [ ] Record the B3LYP open-shell auto-fallback to `LIBXC=HYB_GGA_XC_B3LYP` (spec note) in
      the `Result` metadata so the actual route is captured per calculation.

---

## Phase 4 — ESP charges and Molden surfacing

- [ ] **ESP charges** (`ESP_CHARGE`): spec marks these `[FILE]/[FIELD]` — clean in-memory
      return needs a small binding (`[DEV]`-ish). Scope: expose `esp_charges` property; decide
      whether to read the field directly or parse the `.vdw`/output. Expose `esp_charges`
      on `Result`.
- [ ] **Molden export**: QUICK already writes Molden (`quick_api_module.f90`). Surface it as
      a `Calculation` toggle / `Result.write_molden(path)` rather than re-implementing.

---

## Phase 5 — input conveniences (consistent with current style)

Geometry is passed to the `get_energy`/`get_grad`/`geo_opt` methods as the existing string
form. Add helpers that *produce* that string; do not introduce a `Molecule` class.

- [ ] `geometry_from_xyz_file("mol.xyz")` — read an `.xyz` into the geometry string.
- [ ] Optional RDKit helper (SMILES → 3D coords → geometry string), kept as an optional
      dependency and a thin convenience, per the design doc's RDKit example.

---

## Phase 6 — deferred / needs new Fortran instrumentation (`[DEV]`)

Scope as separate, optional tasks — **not** part of the core deliverable (spec §5):

- [ ] In-memory MO **coefficients** (today only via Molden file).
- [ ] Fine energy split: kinetic vs nuclear–electron attraction (folded into `E1e`);
      Coulomb `J` vs exchange `K` (consumed in the Fock build, not retained).
- [ ] Two-electron energy: can be **derived** (`≈ EEl − E1e − Exc`) as a cheap interim —
      verify bookkeeping in `getEnergy.f90` before relying on it.

---

## Testing strategy (cross-cutting, from design §7)

- [ ] pytest unit test per public function/property.
- [ ] Reference-value checks: every property matches QUICK's own output within the harness
      tolerance (energy ≈ `4.0e-5` Ha as in the existing smoke test).
- [ ] e2e on 3–5 small molecules across HF / DFT / (later) MP2.
- [ ] Cross-validation vs PySCF and Psi4 for energies/gradients/charges.
- [ ] CI: extend the serial workflow already running the pyquick smoke test.

## Out of scope (explicitly)

- `BatchRunner` and any batch orchestration.
- JSON / ASE-database serializers and combined-dataset output.
- YAML batch config.
- GUI, cloud deployment, SLURM submission.
- MPI / GPU / multi-GPU enablement (design §8, future work).
