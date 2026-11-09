"""
Which pyquick mode the tests use.  Set PYQUICK_MODE to serial (the default),
mpi, gpu or gpu_mpi; every test file imports pyquick through import_pyquick(),
so one suite runs in each mode:

    python -m pytest test_pyquick_api.py
    PYQUICK_MODE=mpi mpiexec -n 2 python -m mpi4py -m pytest test_pyquick_api.py
    PYQUICK_MODE=gpu python -m pytest test_pyquick_api.py
    PYQUICK_MODE=gpu_mpi mpiexec -n 2 python -m mpi4py -m pytest test_pyquick_api.py

A mode the install was not built with skips the file.
"""

import importlib
import os

MODULES = {
    'serial':  'pyquick',
    'mpi':     'pyquick.mpi',
    'gpu':     'pyquick.gpu',
    'gpu_mpi': 'pyquick.gpu_mpi',
}


def mode():
    """The mode named by PYQUICK_MODE ('serial' when unset)."""
    name = os.environ.get('PYQUICK_MODE', 'serial')
    if name not in MODULES:
        raise ValueError(f"PYQUICK_MODE={name!r}: expected one of {', '.join(MODULES)}")
    return name


def module_name():
    """The module a script imports for this mode, e.g. 'pyquick.mpi'."""
    return MODULES[mode()]


def import_pyquick():
    """Import pyquick in this mode; skip the calling test file if it is missing."""
    import pytest

    name = module_name()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == 'pyquick':
            pytest.skip("pyquick is not installed (is quick.rc sourced?)",
                        allow_module_level=True)
        raise
    except ImportError as e:
        if 'not available' in str(e):
            pytest.skip(f"PYQUICK_MODE={mode()}: {e}", allow_module_level=True)
        raise
