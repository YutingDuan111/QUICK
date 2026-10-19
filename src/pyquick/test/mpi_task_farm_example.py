"""
Task farming with the SERIAL pyquick build: each MPI rank runs its own,
independent calculations; mpi4py only splits the list and gathers the results.

    mpiexec -n 4 python -m mpi4py mpi_task_farm_example.py

Use the serial build (the default `cmake -DPYTHON=TRUE`), NOT the MPI build.
In the MPI build every calculation is shared by all ranks, so ranks running
different molecules at the same time would wait on each other forever.
"""

from mpi4py import MPI

import pyquick

if getattr(pyquick, "mpi_enabled", lambda: False)():
    raise SystemExit("task farming needs the serial pyquick build, not the MPI build")

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

WATER = "O 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24"
BASES = ["STO-3G", "3-21G", "6-31G", "6-31G*"]

mine = BASES[rank::size]                      # round-robin share for this rank
local = {}
for basis in mine:
    calc = pyquick.PyQuick(method="HF", basis=basis, log=False)
    local[basis] = calc.get_energy(WATER).total_energy

gathered = comm.gather(local, root=0)         # list of dicts, on rank 0 only
if rank == 0:
    energies = {b: e for part in gathered for b, e in part.items()}
    for basis in BASES:
        print(f"RESULT {basis} {energies[basis]:.10f}")
