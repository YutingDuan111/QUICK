"""
pyquick with the serial engine (libquick).

Same engine as plain `import pyquick`, but allowed under mpiexec: every rank
then runs its own serial calculations (task farming).
"""

from . import _api

_api._bind('serial')

from ._api import *
from ._api import __all__


def __getattr__(name):
    # private helpers of the shared API, used by the tests
    if name.startswith('_') and not name.startswith('__') and hasattr(_api, name):
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
