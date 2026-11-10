pyquick
=======

A Python interface to QUICK. pyquick runs QUICK calculations from Python and returns the
results in memory as floats and NumPy arrays, with no output files to parse. It is built
from QUICK's own source with f2py, and it runs on CPUs, with MPI, and on GPUs.

```python
import pyquick

calc = pyquick.PyQuick(method='B3LYP', basis='6-31G*')
result = calc.get_energy("""
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
""")
print(result.total_energy)        # Hartree
```

Features
--------
* Single-point energies, gradients and geometry optimizations
* Hartree-Fock and DFT, restricted and unrestricted
* The energy and each of its components: nuclear repulsion, one-electron (kinetic and
  nuclear attraction), two-electron, exchange-correlation and dispersion
* Molecular orbitals, densities and the overlap matrix
* Mulliken, Löwdin and ESP charges, and the dipole moment
* External point charges, with the gradient on the charges
* Optimization trajectories
* Persistent jobs: a series of geometries of one molecule, each reusing the previous density
* Geometries as strings or `.xyz` files
* Serial, MPI, GPU and multi-GPU engines from one install, chosen by the import

Limitations
-----------
* QUICK's own limitations apply (see the QUICK [README](../../README.md))
* A process uses one engine; switching between CPU and GPU needs a new process
* Most errors inside QUICK end the Python process (or the Jupyter kernel), because QUICK
  stops by calling `exit()`. pyquick checks the input it can before QUICK runs, and with
  logging on the reason is written to the log file

Installation
------------

### Requirements

* Everything QUICK needs (see the [Installation Guide](https://quick-docs.readthedocs.io/en/latest/installation-guide.html#installation-guide))
  and CMake >= 3.12
* Python 3.10 or later, **with development headers**. A venv made from a system Python
  without headers fails in meson with "Python dependency not found".
* For that Python: `numpy >= 2.0`, `meson` and `ninja`.
  ```bash
  python3 -m pip install "numpy>=2.0" meson ninja
  ```
* MPI engines: `mpi4py`, built from source with the same MPI that QUICK was built with:
  ```bash
  MPICC=$(which mpicc) python3 -m pip install --no-binary=mpi4py mpi4py
  ```
* GPU engines: a CUDA Toolkit (NVIDIA) or ROCm (AMD) that QUICK supports.

### Build and install

pyquick is built with QUICK's CMake build: add `-DPYTHON=TRUE`. One build makes every
engine its options allow. pyquick needs the CMake build; the legacy `configure` build does
not build it.

| Engine | Built with | Bound to |
|---|---|---|
| serial | always | `libquick` |
| mpi | `-DMPI=TRUE` | `libquick_mpi` |
| gpu | `-DCUDA=TRUE` or `-DHIP=TRUE` | `libquick_cuda` / `libquick_hip` |
| gpu_mpi | both of the above | `libquick_mpi_cuda` / `libquick_mpi_hip` |

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DMPI=TRUE -DCUDA=TRUE -DQUICK_USER_ARCH=<arch> \
    -DPYTHON=TRUE -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --parallel $(nproc)
cmake --install .
source ../install/quick.rc
```

The engines are
installed in `<install>/lib/pyquick/_engines/<engine>/`, and `quick.rc` puts `<install>/lib`
on `PYTHONPATH`.


### Testing

`runtest --python` runs `test/pyquick_test.py`, a smoke test followed by the pytest. The engine matches each build type: `--serial` in serial, `--mpi` and
`--cudampi`/`--hipmpi` under `DO_PARALLEL`. A build type whose engine was not built is not
run. Without pytest only the smoke test runs.

```bash
cd install
./runtest --serial --python
DO_PARALLEL="mpirun -np 2" 
./runtest --mpi --python
```

The summary shows `Python tests: 0/1 (failed/total)`; the test output is in
`runtest-verbose.log`. 

Choosing an engine
------------------
The import picks the engine. Every engine has the same API.

| Engine | Import | Run with |
|---|---|---|
| serial | `import pyquick` | `python3 script.py` |
| mpi | `import pyquick.mpi as pyquick` | `mpiexec -n 4 python3 -m mpi4py script.py` |
| gpu | `import pyquick.gpu as pyquick` | `python3 script.py` |
| gpu_mpi | `import pyquick.gpu_mpi as pyquick` | `mpiexec -n 2 python3 -m mpi4py script.py` |

* **One engine per process.** Importing a
  second engine raises `ImportError: pyquick is already using the mpi engine in this process`.
* `pyquick.mpi_enabled()`, `gpu_enabled()`, `mpi_rank()` and `mpi_size()` report the engine
  in use (`False`, `False`, `0`, `1` for serial).

Using the interface
-------------------

### Settings: `PyQuick`

```python
calc = pyquick.PyQuick(method, basis, properties=(), charge=0, mult=1,
                       keywords=None, log=True)
```

* `method`: `'HF'` or `'UHF'`, or a DFT functional (`'B3LYP'`, `'PBE0'`, `'BLYP'`, ...).
  `mult > 1` selects the unrestricted path (UHF/UDFT).
* `basis`: a QUICK basis set name, e.g. `'STO-3G'`, `'6-31G*'`.
* `properties`: extras to compute: `'mulliken_charges'`, `'lowdin_charges'`, `'dipole'`,
  `'esp_charges'`.
* `charge`, `mult`: total charge and spin multiplicity. Impossible combinations raise
  `ValueError` before QUICK runs.
* `keywords`: extra QUICK keywords. `{'cutoff': '1.0e-9'}` becomes `CUTOFF=1.0e-9`, and a value
  of `None` gives a bare keyword, e.g. `{'D3': None}`, `{'LOPT': None}` (in-house optimizer).
* `log`: write QUICK's log to `<name>.out` (`pyquick_job.out` without a name). With
  `log=False` nothing is written to disk.

One `PyQuick` object can run many calculations.

### Calculations

The method you call chooses the job type, and each returns a `Result`.

```python
result = calc.get_energy(geometry)                 # single-point energy
result = calc.get_grad(geometry)                   # energy and gradient
result = calc.geo_opt(geometry, max_cycles=None)   # geometry optimization
```

* `geometry`: a `'SYMBOL X Y Z'` string, one atom per line, in Angstrom, or the path of an
  `.xyz` file (`'mol.xyz'` or a `pathlib.Path`). `pyquick.read_xyz(path)` returns a file's
  geometry as a string.
* `name='water'` labels the run and names its log file (`water.out`).
* `point_charges=[(x, y, z, q), ...]` adds external point charges (Angstrom, atomic units of
  charge) for this call only.
* `geo_opt` uses DL-FIND by default and needs at least 3 atoms; add `{'LOPT': None}` to the
  keywords for the in-house optimizer. Without `max_cycles` it stops after 2000 steps.

### Results: `Result`

| Attribute | Available | Units |
|---|---|---|
| `total_energy`, `nuclear_repulsion`, `e_electronic` | always | Hartree |
| `e_one_electron`, `e_kinetic`, `e_nuclear_attraction`, `e_two_electron`, `e_xc` | always | Hartree |
| `e_dispersion` | with a `D3`/`D3BJ` keyword | Hartree |
| `e_external_charge` (charge-charge repulsion, not in `total_energy`) | with `point_charges` | Hartree |
| `atomic_numbers`, `coordinates` | always | Angstrom |
| `mo_energies`, `mo_coefficients`, `mo_occupations`, `density_matrix`, `overlap_matrix`, `ao_labels` | always | Hartree, AO basis |
| `beta_mo_energies`, `beta_mo_coefficients`, `beta_mo_occupations`, `beta_density_matrix` | unrestricted runs | |
| `gradient` `(natom, 3)` | `get_grad` | Hartree/Bohr |
| `point_charge_gradient` `(ncharges, 3)` | `get_grad` with `point_charges` | Hartree/Bohr |
| `optimized_coordinates`, `converged`, `trajectory_energies`, `trajectory_coordinates` | `geo_opt` | Angstrom, Hartree |
| `mulliken`, `lowdin`, `dipole`, `esp_charges` | when requested in `properties` | e, Debye |

* `total_energy = nuclear_repulsion + e_electronic (+ e_dispersion)`, and
  `e_electronic = e_one_electron + e_two_electron + e_xc`.
* On restricted runs `density_matrix` is the total density; on unrestricted runs it is the
  alpha density (total = `density_matrix + beta_density_matrix`).

### Persistent jobs: `Job`

A job keeps one molecule set up in QUICK. Each step reuses the previous density, so a series
of geometries (a scan, an MD trajectory) converges faster than separate calculations.

```python
with calc.new_job(geometries[0]) as job:
    for geom in geometries:
        result = job.run(geom, gradient=True)
```

The atoms must stay the same; only the coordinates change. `point_charges` given to
`new_job` apply to every step.

Running with MPI
----------------
```bash
mpiexec -n 4 python3 -m mpi4py my_script.py
```

* Run through `python3 -m mpi4py`, so that an uncaught error stops every rank instead of
  leaving the others waiting.
* Every rank runs the whole script and gets the same `Result`; only rank 0 writes the log.
  Print from one rank with `if pyquick.mpi_rank() == 0: ...`.
* An error inside QUICK stops all ranks (`MPI_Abort`); input errors raise on every rank.
* Without `mpiexec` the script runs as one rank. Jupyter is not supported for MPI runs.
* Across several nodes, pass the environment variables your script needs with `mpiexec -x`.

**Task farming** runs independent calculations on different ranks with the serial engine:

```python
from mpi4py import MPI
import pyquick.serial as pyquick

comm = MPI.COMM_WORLD
water = "O 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24"
bases = ["STO-3G", "3-21G", "6-31G", "6-31G*"]

mine = {b: pyquick.PyQuick('HF', b, log=False).get_energy(water).total_energy
        for b in bases[comm.Get_rank()::comm.Get_size()]}
parts = comm.gather(mine, root=0)
if comm.Get_rank() == 0:
    print({b: e for part in parts for b, e in part.items()})
```

Running on GPUs
---------------
`pyquick.gpu` uses one GPU, chosen with `CUDA_VISIBLE_DEVICES`. `pyquick.gpu_mpi` gives each
MPI rank a GPU, as `quick.cuda.MPI` does; any MPI library works.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python3 -m mpi4py my_script.py
```
