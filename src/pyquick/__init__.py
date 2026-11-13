"""
pyquick: Python interface to QUICK.

The import picks the engine:

    import pyquick                  serial (loads on first use)
    import pyquick.mpi as pyquick   MPI: one calculation on all ranks
    import pyquick.gpu as pyquick   one GPU
    import pyquick.gpu_mpi as pyquick   one GPU per MPI rank
    import pyquick.serial as pyquick    serial, also under mpiexec (task farming)

Every engine has the same API: PyQuick, Result, Job, read_xyz and the
mpi_* / gpu_* queries.  A process can use only one engine.  An engine the
install was built without raises ImportError when imported.
"""

import os as _os

from . import _api
from ._api import read_xyz

__all__ = _api.__all__

# Environment variables an MPI launcher sets to the number of ranks.
_LAUNCHER_SIZE = ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE')


def _launcher_ranks():
    """Ranks of the mpiexec job this process belongs to (1 outside mpiexec)."""
    for name in _LAUNCHER_SIZE:
        try:
            return int(_os.environ[name])
        except (KeyError, ValueError):
            pass
    return 1


def _bind_serial():
    """Plain `import pyquick`: the serial engine, refused under mpiexec.

    An MPI script that forgot pyquick.mpi would otherwise run the whole
    calculation once on every rank, each on one core.  pyquick.serial skips
    this check for scripts that want that (task farming).
    """
    if _api._MODE is None:
        nranks = _launcher_ranks()
        if nranks > 1:
            raise ImportError(
                f"running under mpiexec with {nranks} ranks, but import pyquick is the "
                f"serial engine; use import pyquick.mpi (or pyquick.gpu_mpi), or "
                f"import pyquick.serial to run serial on every rank")
    _api._bind('serial')


def __getattr__(name):
    # Module-level __getattr__ (Python 3.7+): the engine loads on first use of
    # the API, so `import pyquick.mpi` (which imports this package first)
    # never loads the serial engine.
    if name in _api.__all__:
        _bind_serial()
        return getattr(_api, name)
    if name.startswith('_') and not name.startswith('__') and hasattr(_api, name):
        return getattr(_api, name)          # private helpers, used by the tests
    raise AttributeError(f"module 'pyquick' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_api.__all__))
