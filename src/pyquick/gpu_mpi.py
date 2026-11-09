"""
pyquick with the GPU + MPI engine (libquick_mpi_cuda or _hip): one GPU per rank.

    mpiexec -n 2 python -m mpi4py script.py

Rank r uses visible GPU r % count.  Needs mpi4py, built against the same MPI
library as QUICK.
"""

from . import _api

_api._bind('gpu_mpi')

from ._api import *  # noqa: E402,F401,F403
from ._api import __all__  # noqa: E402,F401


def __getattr__(name):
    # private helpers of the shared API, used by the tests
    if name.startswith('_') and not name.startswith('__') and hasattr(_api, name):
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
