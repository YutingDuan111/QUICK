"""
pyquick with the MPI engine (libquick_mpi): one calculation on all ranks.

    mpiexec -n 4 python -m mpi4py script.py

Needs mpi4py, built against the same MPI library as QUICK.
"""

from . import _api

_api._bind('mpi')

from ._api import *  # noqa: E402,F401,F403
from ._api import __all__  # noqa: E402,F401


def __getattr__(name):
    # private helpers of the shared API, used by the tests
    if name.startswith('_') and not name.startswith('__') and hasattr(_api, name):
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
