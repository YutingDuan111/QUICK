"""
Task farming with the SERIAL pyquick engine: each MPI rank runs its own,
independent calculations; mpi4py only splits the list and gathers the results.

    mpiexec -n 4 python -m mpi4py mpi_task_farm_example.py

Import pyquick.serial, not plain pyquick (which stops under mpiexec, to catch
MPI scripts that forgot pyquick.mpi) and not pyquick.mpi (where every
calculation is shared by all ranks, so ranks running different molecules at
the same time would wait on each other forever).
"""

from mpi4py import MPI

import pyquick.serial as pyquick

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
