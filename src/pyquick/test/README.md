# pyquick tests

Run them from this directory, after `source <install>/quick.rc`. Each pytest suite runs in
the mode named by `PYQUICK_MODE` (`serial` by default, `mpi`, `gpu` or `gpu_mpi`; see
`pyquick_test_mode.py`). A mode the install was not built with skips the suite.

## Files
- `test_pyquick_modes.py` — how a script picks an engine: which engines exist, lazy loading,
  one engine per process, the `mpiexec` guard on plain `import pyquick`, the same energy in
  every mode. Plain pytest; every case starts its own process.
- `test_pyquick_api.py` — the API: construction and validation, energy-model identities,
  `Result` gating, geometry access, properties, gradients, optimizations, persistent jobs.
- `test_pyquick_xyz.py` — geometries from `.xyz` files.
- `test_pyquick_mpi.py` — MPI modes only, run under `mpiexec`: identical results on every
  rank, rank 0 writes the files, errors leave no rank behind.
- `test_pyquick_gpu_mpi.py` — `gpu_mpi` only, under `mpiexec`: one GPU per rank, energies
  against the `quick` binary references.
- `test_pyquick_mpi_launch.py` — MPI modes only, plain pytest: starts `mpiexec` itself (rank
  counts, single rank, engine errors, task farming).
- `mpi_task_farm_example.py` — task farming with `pyquick.serial`.
- `../../../test/pyq_ene_H2O_rhf_sto3g.py` — the water energy smoke test that `runtest` runs in
  each mode.

## Prerequisites
1. Build with `-DPYTHON=TRUE` for the Python you'll test with, plus `-DMPI=TRUE` and/or
   `-DCUDA=TRUE` for those modes.
2. `pip install pytest`, and mpi4py for the MPI modes (built against QUICK's MPI; see the build
   guide).

## Run
```sh
python3 -m pytest test_pyquick_modes.py -q
python3 -m pytest test_pyquick_api.py test_pyquick_xyz.py -q
PYQUICK_MODE=gpu python3 -m pytest test_pyquick_api.py test_pyquick_xyz.py -q
PYQUICK_MODE=mpi mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_api.py test_pyquick_xyz.py test_pyquick_mpi.py -q
PYQUICK_MODE=mpi python3 -m pytest test_pyquick_mpi_launch.py -q
PYQUICK_MODE=gpu_mpi mpiexec -n 2 python3 -m mpi4py -m pytest test_pyquick_gpu_mpi.py -q

# the smoke test, in any mode
PYQUICK_MODE=mpi mpiexec -n 2 python3 -m mpi4py ../../../test/pyq_ene_H2O_rhf_sto3g.py
```

`PYQUICK_MPIEXEC` sets the launcher for the files that start `mpiexec` themselves (for example
`"mpiexec --oversubscribe"` when ranks > cores).
