# QUICK Python API — Parameter & Result Specification

Grounded in the current `merzlab/QUICK` source (`quick_method_module.f90`,
`quick_molspec_module.f90`, `quick_calculated_module.f90`, `quick_api_module.f90`).
Defaults and keyword spellings reflect that source — re-verify against your build before
freezing the API.

Legend for availability:
**[API]** already returned by the library API · **[FIELD]** stored as a named field, needs a binding ·
**[FILE]** written to a file (parse/read it) · **[DERIVE]** computable from stored fields ·
**[DEV]** not stored separately, needs new instrumentation in the Fortran layer.

---

## 1. Setup (input) parameters

### 1a. Required — user must always provide

| Parameter | Notes |
|---|---|
| **Geometry** | element symbols + Cartesian coordinates (or atomic numbers + coords). For the library API: `atomic_numbers` at job setup, `coords (3, natoms)` per call. |
| **Basis set** | `BASIS=...` — **no default**; QUICK errors without it. |
| **Method (Hamiltonian)** | `HF`, `DFT` + a functional, or `MP2`. No implicit default — specify one. |

### 1b. Optional — has a default

| Parameter | Keyword | Default | Meaning |
|---|---|---|---|
| Charge | `CHARGE=` | `0` | total molecular charge |
| Multiplicity | `MULT=` | `1` (singlet) | spin multiplicity; >1 ⇒ unrestricted path |
| Job type | `ENERGY` / `GRADIENT` / `OPTIMIZE` | `ENERGY` | single-point if none given |
| SCF max cycles | `NCYC=` | `200` | max SCF iterations |
| Density convergence | `DENSERMS=` | `1.0e-6` | density-matrix RMS convergence |
| Integral cutoff | `CUTOFF=` | `1.0e-7` | 2e-integral screening |
| Gradient cutoff | `GRADCUTOFF=` | `1.0e-7` | gradient integral screening |
| Density-matrix cutoff | — | `1.0e-10` | DM screening |
| XC cutoff | `XCCUTOFF=` | `1.0e-7` | exchange-correlation grid screening |
| Max DIIS | `MAXDIIS=` | `10` | DIIS subspace size |
| DFT grid | `SG0` / `SG1` | `SG1` | XC integration grid (SG0 = coarse) |
| Initial guess | `RDSAD` / `WRSAD` | `SAD` | superposition of atomic densities |
| Dispersion | `D2`/`D3`/`D3BJ`/`D3M`/`D3MBJ` | none | empirical dispersion correction |
| ESP grid spacing | `ESPGRID_SPACING=` | `0.25` Å | only with `ESP_CHARGE` |

> Note: `TIGHTINT` / `COARSEINT` are convenience presets that tighten/loosen several
> cutoff + convergence values at once (TIGHT ≈ `1e-8`/`1e-7`, COARSE ≈ `1e-6`/`1e-5`).

### 1c. Property / output toggles — off by default, opt-in

| Toggle | Keyword | Produces |
|---|---|---|
| Dipole moment | `DIPOLE` | dipole vector + magnitude |
| ESP charges | `ESP_CHARGE` | ESP-fitted atomic charges (+ `.vdw` grid file) |
| Molden export | `MOLDEN` | `<base>.molden.<step>` (geometry, basis, MOs, SCF/opt trajectory) |
| Write density | `CHK_WRITE` / `CHK_WRITE_DEN` | density matrix → `<base>.dat` (HDF5 if built with `RESTART_HDF5`) |
| Read density | `CHK_READ_DEN` | restart SCF from a saved density |
| External charges | `EXTCHARGES` | QM energy/gradient in a field of point charges |

---

## 2. Value lists for key parameters

**Method / Hamiltonian:** `HF`, `DFT`, `MP2`.

**DFT functionals** (native + Libxc aliases; see note):
`B3LYP`, `BLYP`, `PBE`, `PBE0`, `revPBE`, `BP86`, `PW91`, `OLYP`, `O3LYP`, `B97`,
`MPW91LYP`, `MPW91PW91`, or an explicit `LIBXC=<id>` such as
`LIBXC=HYB_GGA_XC_B3LYP` or `LIBXC=GGA_X_B88,GGA_C_LYP` (max two functionals).

> Most named functionals are internally Libxc aliases. `B3LYP` is special: it uses
> QUICK's **native closed-shell** implementation, but **auto-falls back to
> `LIBXC=HYB_GGA_XC_B3LYP` for open-shell** systems. Record the actual route per
> calculation — the two paths can give slightly different energies.

**Basis sets** (examples — many available in the basis directory):
`STO-3G`, `3-21G`, `6-31G`, `6-31G*`, `6-31G**`, `6-311G**`, `cc-pVDZ`, `cc-pVTZ`,
`def2-SVP`, `def2-TZVP`.

**Job type:** `ENERGY` (default), `GRADIENT`, `OPTIMIZE`.

**Dispersion:** `D2`, `D3`, `D3BJ`, `D3M`, `D3MBJ`.

**Charge:** any integer. **Multiplicity:** positive integer (`1` = singlet, `2` = doublet, …).

---

## 3. Results available by setup combination

Each row is cumulative with the rows above it.

| Setup | Results that become available |
|---|---|
| **`ENERGY`** (single point) | total energy + energy components (§4); molecular orbitals; density matrix; dipole *(if `DIPOLE`)*; Mulliken & Löwdin charges; ESP charges *(if `ESP_CHARGE`)* |
| **`GRADIENT`** | all of the above **+ nuclear gradients (forces)**, shape `(natoms, 3)`, Hartree/Bohr |
| **`OPTIMIZE`** | all of the above **+ optimized coordinates**; **optimization trajectory** (per-step geometries + energies, via Molden `[GEOMETRIES]`/`[GEOCONV]`) |
| **`+ EXTCHARGES` with `GRADIENT`** | **+ forces on the external point charges** (`ptchg_grad`) |
| **`+ MOLDEN`** (any of the above) | a portable `.molden` file carrying geometry + basis + MO coefficients (+ trajectory for OPT) |

---

## 4. Useful properties — availability & where they live

| Property | Status | Source / how to expose |
|---|---|---|
| **Energy (total)** | **[API]** | `getQuickEnergy` / `getQuickEnergyGradients` → `ETot` |
| **Nuclear gradients (forces)** | **[API]** | `getQuickEnergyGradients` → `gradients(3,natoms)` |
| **Dipole moment** | **[FIELD]** | needs `DIPOLE`; bind the computed dipole vector |
| **Mulliken charges** | **[FIELD]** | `quick_qm_struct%Mulliken(natom)` — bind it |
| **Löwdin charges** | **[FIELD]** | `quick_qm_struct%Lowdin(natom)` — bind it |
| **ESP charges** | **[FILE]/[FIELD]** | needs `ESP_CHARGE`; charges in output, grid in `.vdw` — bind or parse |
| **Molecular orbitals** | **[FILE]** | `MOLDEN` export (`exportMO`): MO coeffs + energies + occupations. No clean in-memory return yet ⇒ **[DEV]** if you want it without the file |
| **Density matrix** | **[FIELD]** | live in-memory arrays `quick_qm_struct%dense`/`denseb`; harvest via f2py as NumPy `float64` (see density-matrix note below) |
| **Optimized coordinates** | **[FIELD]/[FILE]** | final geometry from the OPT job; per-step trajectory in Molden |

> **Density-matrix storage.** The density matrix is *not* re-read from QUICK's binary
> `.dat`/HDF5 restart file. pyquick harvests the in-memory `quick_qm_struct%dense`
> (and `denseb` for open shell) directly through f2py, so `Result.density_matrix` is a
> NumPy `float64` array of shape `(nbasis, nbasis)` — no opaque binary blob, no restart-file
> parsing. The API does not serialize it; users store it themselves (e.g. an `h5py`
> dataset with `compression='gzip'`).

---

## 5. Energy components — availability

QUICK stores these as **named scalar fields** in `quick_qm_struct` (directly bindable):

| Component | Field | Status |
|---|---|---|
| Total energy | `ETot` | **[API]** |
| Electronic energy | `EEl` | **[FIELD]** |
| Nuclear repulsion (core–core) | `ECore` | **[FIELD]** ← your "Ecore" |
| One-electron energy | `E1e` | **[FIELD]** = kinetic + nuclear–electron attraction (**combined**) |
| Exchange–correlation energy | `Exc` | **[FIELD]** |
| Dispersion correction | `Edisp` | **[FIELD]** (if a `D*` keyword set) |
| External point-charge energy | `ECharge` | **[FIELD]** (if `EXTCHARGES`) |
| MP2 correlation | `EMP2` | **[FIELD]** (if `MP2`) |

### pyquick attribute names (naming decision)

The QUICK field `ECore` is the **core–core nuclear repulsion**; its "core" name is
ambiguous, so the API renames it. Final `Result` attribute names:

| QUICK field | `Result` attribute | Meaning | Availability |
|---|---|---|---|
| `ETot` | `total_energy` | total energy | always |
| `ECore` | `nuclear_repulsion` | core–core nuclear repulsion (**renamed** from `ECore`/`e_core`) | always |
| `EEl` | `e_electronic` | electronic energy | always |
| `E1e` | `e_one_electron` | kinetic + nuclear–electron attraction (combined) | always |
| — (derived) | `e_two_electron` | J + K combined (`EEl − E1e − Exc`); **J and K not exposed individually** | always |
| `Exc` | `e_xc` | DFT exchange-correlation (exchange + correlation combined); 0 for HF | always |
| `Edisp` | `e_dispersion` | empirical dispersion; **only when a `D*` keyword is requested**, and only then in `total_energy` | conditional |
| `ECharge` | `e_external_charge` | QM↔external-charge energy; **never in `total_energy`**, reported only when `EXTCHARGES` is set | conditional |

Model: `total_energy = nuclear_repulsion + e_electronic (+ e_dispersion if requested)`,
with `e_electronic = e_one_electron + e_two_electron + e_xc`.

### Exact decomposition (traced through the source)

```
Etot = Eel + Ecore  (+ Edisp if dispersion)        # getEnergy.f90:152, :157
Eel  = E1e + E2e + Exc   (Exc = 0 for HF)           # getCshellEriEnergy + :528
E1e  = Tr(P · Hcore)                                 # oei_module:55-56 (stored)
⟹  E2e = Eel − E1e − Exc                            # bookkeeping-exact
```

`Eel` is formed as `½ Σ Pij(Fji + Hcoreji) = E1e + E2e`, then `Exc` is added for DFT.
`E1e` is independently stored as `Tr(P·Hcore)`, so the two-electron term is an exact
subtraction, not an approximation.

| Component | Status | Note |
|---|---|---|
| Two-electron energy | **[DERIVE]** | `EEl − E1e − Exc`, exact. **Label carefully** — see method note below |
| Electron kinetic energy (alone) | **[DEV]** | inside `E1e`; KE integrals built separately (`kineticO`) ⇒ recoverable as `Tr(P·KE)` with modest work if the KE matrix is retained |
| Nuclear–electron attraction (alone) | **[DEV]** | inside `E1e`; built by `attrashell`; same situation |
| Coulomb J (alone) | **[DEV]** | formed with K inside the ERI build, never stored — harder |
| Exchange K (alone) | **[DEV]** | same |

**Method-dependent content of E2e** (don't label it "Coulomb"):
- HF: `E2e = J + K` (full exact exchange), `Exc = 0`
- Pure DFT: `E2e = J`; all exchange in `Exc`
- Hybrid (B3LYP 20%): `E2e = J + 0.2·K_exact`; rest in `Exc`

Safe label for the derived term: *two-electron energy (Coulomb + exact-exchange admixture)*.

**Takeaway:** the coarse decomposition (total, electronic, nuclear repulsion, one-electron,
two-electron, XC) is essentially free — stored fields plus one exact subtraction. The *fine*
split (kinetic vs attraction; J vs K) needs instrumentation in the SCF operator routines.
Kinetic/attraction is the easier half (separate integral routines exist); J/K is harder.
Scope the fine split as a separate, optional task, not a v1 binding.

> **EXTCHARGES decision:** `getEnergy.f90:150` adds `ECharge` to `Etot`, but `:152` then
> overwrites `Etot = Eel + Ecore`, so `ECharge` is dropped from the total as written. This is
> the intended behavior for the API: **`e_external_charge` is never part of `total_energy`**.
> The API still exposes `e_external_charge` as a separate reported value whenever `EXTCHARGES`
> is set (bind `quick_qm_struct%ECharge` + a `quick_method%extcharges` availability flag).

---

## 6. Suggested minimal v1 vs later

**v1 (bind what already exists):** geometry in; `ENERGY`; return `total_energy` + components
(`nuclear_repulsion`, `e_electronic`, `e_one_electron`, `e_two_electron` by subtraction,
`e_xc`; `e_dispersion`/`e_external_charge` when requested), Mulliken/Löwdin charges, alpha MO
energies, alpha density matrix. These are all stored fields or exact derivations. (`GRADIENT`,
`OPTIMIZE`, and the dipole vector move to Phase 2 — see the development steps.)

**Later (needs Fortran-side work):** in-memory MO coefficients and density matrix (today they
go to files); fine energy split (kinetic/attraction, J/K); ESP charges as a clean return value.

---

## 7. Data formats (codebase → API return type)

How each quantity is represented today in the Fortran layer and what type the Python API
exposes. The API does **not** write files — the "suggested HDF5" column is guidance for users
serializing a dataset themselves (a few `h5py` lines), not an API output schema.

### 7a. Cross-cutting conventions — decide once, apply everywhere

1. **Layout: canonicalize to row-major, don't pass Fortran's through.** QUICK stores
   column-major (`xyz`, `dense` as `(3,natom)`/`(nbasis,nbasis)`; **gradient is a flat
   `(3*natom)` vector**). The API returns Python-natural `(natom, 3)` for per-atom vectors,
   reshaping/transposing at the boundary so no consumer has to guess.
2. **Units: store as an attribute on every numeric dataset, never assume.** Energies =
   Hartree, forces = Hartree/Bohr. Coordinates are ambiguous (Å input vs Bohr internal — the
   Molden export is "AU") and the dipole may be au or Debye. Attach a `units` attribute and
   **verify coordinate + dipole units against the source** before trusting them.
3. **dtype: float64 throughout.** Reaction energies live in the last few digits — don't
   downcast energies or forces. Only consider float32 for the big matrices if storage becomes
   a real constraint.
4. **Attribute vs dataset.** Scalar metadata (method, basis, charge, multiplicity, functional,
   job type, `natom`, `nbasis`) → HDF5 **attributes**. Numeric results (arrays + the energy
   values) → **datasets**.

### 7b. Per-quantity mapping

| Property | Codebase (Fortran) | API (Python) | Suggested HDF5 (user-side) |
|---|---|---|---|
| Energies (`ETot`,`EEl`,`E1e`,`Exc`,`ECore`,`Edisp`,`EMP2`) | `double precision` scalar | `float` | scalar `f8`, `units="Hartree"` |
| Coordinates (`xyz`) | `double (3,natom)` | `ndarray (natom,3) f8` *(transposed)* | `(natom,3) f8`, `units=?` ← verify Å vs Bohr |
| Elements (`iattype`) | `integer (natom)` | `ndarray (natom,) i4` (atomic #) + symbol list | `(natom,) i4` |
| Forces (`gradient`) | `double (3*natom)` **flat** | `ndarray (natom,3) f8` *(reshaped)* | `(natom,3) f8`, `units="Hartree/Bohr"` |
| Mulliken / Löwdin charges | `double (natom)` | `ndarray (natom,) f8` | `(natom,) f8` |
| ESP charges | computed → file | `ndarray (natom,) f8` | `(natom,) f8` |
| Dipole | **not stored** (local in `dipole.f90`) | `ndarray (3,) f8` + magnitude | `(3,) f8`, `units=?` ← verify au vs Debye |
| MO coefficients (`co`, `cob`) | `double (nbasis,nbasis)` | `ndarray (nbasis,nbasis) f8` | `(nbasis,nbasis) f8`, **chunked + gzip** |
| Orbital energies (`E`, `Eb`) | `double (nbasis)` | `ndarray (nbasis,) f8` | `(nbasis,) f8`, `units="Hartree"` |
| Occupations | from `iDegen`/`aElec`/`bElec` | `ndarray (nbasis,) f8` | `(nbasis,) f8` |
| Density matrix (`dense`, `denseb`) | `double (nbasis,nbasis)` | `ndarray (nbasis,nbasis) f8` | `(nbasis,nbasis) f8`, **chunked + gzip** |
| Opt trajectory (`opt_traj`) | `double (3,natom,nsteps)` extendable | `ndarray (nsteps,natom,3) f8` | `(nsteps,natom,3) f8`, chunked |
| `natom`, `nbasis` | `integer` | `int` | **attribute** |
| method/basis/charge/mult/functional/jobtype | parsed str/int | `str`/`int` | **attribute** |

### 7c. HDF5 layout notes (for users serializing themselves)

The API returns plain floats/NumPy arrays; the following is *suggested* guidance if a user
chooses `h5py` — nothing here is produced by the API.

- **Compress only the big matrices.** Density and MO coefficients are `nbasis²` and dominate
  file size — store them chunked with gzip. Everything else is small; plain datasets are fine.
  (This is also the concrete reason these never belong in JSON/CSV.)
- **Suggested group structure** (mirrors the availability tiers in §3):
  `/geometry` (coords, elements) · `/energies` (scalar components) ·
  `/properties` (charges, dipole, forces) · `/wavefunction` (MOs, density),
  with metadata as root-level attributes.
- **Crash-safe batch resume** is a user-side `if os.path.exists(path): continue` around the
  per-molecule write — no API support needed.
