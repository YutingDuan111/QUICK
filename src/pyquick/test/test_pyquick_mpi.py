"""
MPI-specific tests for pyquick, run INSIDE mpiexec with the MPI build
(cmake -DMPI=TRUE -DPYTHON=TRUE):

    mpiexec -n 2 python -m mpi4py -m pytest test_pyquick_mpi.py -q
    mpiexec -n 4 python -m mpi4py -m pytest test_pyquick_mpi.py -q

Every rank runs the whole file.  The module is skipped on a serial or GPU build.

What the existing test_pyquick_api.py (also run under mpiexec) cannot see, and
this file checks:
  * every rank returns the SAME Result, bit for bit (rank 0's values are
    broadcast; workers never read Fortran result state, which is garbage there);
  * QUICK really ran in parallel (its log reports the rank count);
  * only rank 0 writes files, and log=False leaves nothing on any rank;
  * an error raised before QUICK starts is catchable on every rank and the
    next calculation still runs (no rank left waiting).

Rule for every test here: a rank never fails an assertion on its own before
the other ranks reach the same collective.  Checks are reduced across ranks
with _all_ranks_agree() first, so a failure fails on every rank instead of
leaving the others waiting in a later broadcast.
"""

import os
import shutil
import tempfile

import numpy as np

import pytest

pyquick = pytest.importorskip("pyquick")

if not (hasattr(pyquick, "mpi_enabled") and pyquick.mpi_enabled()):
    pytest.skip("pyquick was not built with MPI (-DMPI=TRUE)", allow_module_level=True)

from mpi4py import MPI  # noqa: E402  (only after the skip: serial builds may lack mpi4py)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


WATER = """
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
"""
WATER_STRETCHED = """
    O  -0.33840   0.00380   0.23923
    H  -0.33508  -0.00193  -0.83813
    H   0.67856  -0.00193   0.59530
"""
H2 = "H 0.0 0.0 0.0\nH 0.0 0.0 0.74"
CHARGES = [(5.0, 0.0, 0.0, 1.0), (6.0, 0.0, 0.0, 1.0)]
LOOSE = {'cutoff': '1.0e-9', 'denserms': '1.0e-6'}
TOL = 4.0e-5

# Every public Result attribute.  Ones a run did not produce raise
# AttributeError and are left out of the comparison.
RESULT_FIELDS = (
    'total_energy', 'nuclear_repulsion', 'e_electronic', 'e_one_electron',
    'e_two_electron', 'e_xc', 'e_dispersion', 'e_external_charge',
    'atomic_numbers', 'coordinates', 'mo_energies', 'density_matrix',
    'beta_mo_energies', 'beta_density_matrix', 'mulliken', 'lowdin', 'dipole',
    'esp_charges', 'gradient', 'optimized_coordinates', 'converged',
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _all_ranks_agree(ok):
    """True only if `ok` is true on every rank (a collective: call it on all ranks)."""
    return COMM.allreduce(bool(ok), op=MPI.LAND)


def _fields(result):
    out = {}
    for name in RESULT_FIELDS:
        try:
            out[name] = np.asarray(getattr(result, name))
        except AttributeError:
            pass
    return out


def _identical_to_rank0(result):
    """Compare this rank's Result to rank 0's, field by field, exactly.

    Returns (ok_everywhere, my_problems).  Collective.
    """
    mine = _fields(result)
    root = COMM.bcast(mine if RANK == 0 else None, root=0)
    problems = []
    if set(mine) != set(root):
        problems.append(f"fields differ: {sorted(set(mine) ^ set(root))}")
    for name in set(mine) & set(root):
        if mine[name].shape != root[name].shape or not np.array_equal(mine[name], root[name]):
            problems.append(name)
    return _all_ranks_agree(not problems), problems


@pytest.fixture
def shared_dir(monkeypatch):
    """One directory that EVERY rank works in (pytest's tmp_path differs per rank)."""
    path = tempfile.mkdtemp(prefix="pyquick_mpi_") if RANK == 0 else None
    path = COMM.bcast(path, root=0)
    monkeypatch.chdir(path)
    yield path
    COMM.Barrier()
    if RANK == 0:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# the build and the rank bookkeeping
# ---------------------------------------------------------------------------

def test_rank_and_size_match_mpi4py():
    ok = pyquick.mpi_rank() == RANK and pyquick.mpi_size() == SIZE
    assert _all_ranks_agree(ok), (
        f"rank {RANK}: pyquick says {pyquick.mpi_rank()}/{pyquick.mpi_size()}, "
        f"mpi4py says {RANK}/{SIZE}")


# ---------------------------------------------------------------------------
# every rank gets the same Result, for every job type and option
# ---------------------------------------------------------------------------

def _hf(**kw):
    return pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False, **kw)


def _b3lyp(**kw):
    return pyquick.PyQuick('B3LYP', 'STO-3G', keywords=LOOSE, log=False, **kw)


CASES = {
    'energy':        lambda: _hf().get_energy(WATER),
    'energy_props':  lambda: _hf(properties=['mulliken_charges', 'lowdin_charges',
                                             'dipole']).get_energy(WATER),
    'grad':          lambda: _hf().get_grad(WATER),
    'geo_opt_dlfind': lambda: _hf().geo_opt(WATER),
    'geo_opt_lopt':  lambda: pyquick.PyQuick('HF', 'STO-3G', log=False,
                                             keywords={**LOOSE, 'LOPT': None}).geo_opt(WATER),
    'b3lyp':         lambda: _b3lyp().get_energy(WATER),
    'b3lyp_d3_grad': lambda: pyquick.PyQuick('B3LYP', 'STO-3G', log=False,
                                             keywords={**LOOSE, 'D3': None}).get_grad(WATER),
    'uhf':           lambda: _hf(charge=1, mult=2).get_energy(WATER),
    'ub3lyp_grad':   lambda: _b3lyp(charge=1, mult=2).get_grad(WATER),
    'esp':           lambda: _hf(properties=['esp_charges']).get_energy(WATER),
    'point_charges': lambda: _hf().get_energy(WATER, point_charges=CHARGES),
    'point_charges_grad': lambda: _hf().get_grad(WATER, point_charges=CHARGES),
    # an s-only molecule right after one with p shells (GPU known issue #2)
    'h2_after_water': lambda: (_hf().get_energy(WATER), _hf().get_energy(H2))[1],
}


@pytest.mark.parametrize("case", list(CASES))
def test_every_rank_gets_identical_result(case):
    result = CASES[case]()
    ok, problems = _identical_to_rank0(result)
    assert ok, f"rank {RANK}: differs from rank 0 in {problems}"


@pytest.mark.parametrize("method", ['HF', 'B3LYP'])
def test_persistent_job_identical_on_all_ranks(method):
    calc = pyquick.PyQuick(method, 'STO-3G', keywords=LOOSE, log=False)
    geometries = [WATER, WATER_STRETCHED, WATER]
    with calc.new_job(WATER) as job:
        steps = [job.run(g, gradient=(k == 2)) for k, g in enumerate(geometries)]
    for k, r in enumerate(steps):
        ok, problems = _identical_to_rank0(r)
        assert ok, f"rank {RANK}, step {k}: differs from rank 0 in {problems}"
    # the steps are real: same geometry -> same energy, different -> different
    assert steps[0].total_energy == pytest.approx(steps[2].total_energy, abs=TOL)
    assert abs(steps[0].total_energy - steps[1].total_energy) > 1e-6


# ---------------------------------------------------------------------------
# QUICK really ran on all ranks; files are written by rank 0 only
# ---------------------------------------------------------------------------

def _total_ranks_in_log(path):
    """The rank count from QUICK's '| TOTAL RANKS     =     N' log line, or None."""
    with open(path) as f:
        for line in f:
            if 'TOTAL RANKS' in line:
                return int(line.split('=')[1])
    return None


def test_log_reports_all_ranks(shared_dir):
    pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE).get_energy(WATER, name='mpi_log')
    COMM.Barrier()
    ranks_in_log = _total_ranks_in_log('mpi_log.out') if RANK == 0 else None
    ranks_in_log = COMM.bcast(ranks_in_log, root=0)
    assert ranks_in_log == SIZE, f"log reports {ranks_in_log} ranks, expected {SIZE}"


def test_only_one_log_file(shared_dir):
    """Rank 0 writes <name>.out; workers write nothing (no fort.* files either)."""
    pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE).get_energy(WATER, name='mpi_one_log')
    COMM.Barrier()
    files = sorted(os.listdir(shared_dir))
    assert _all_ranks_agree(files == ['mpi_one_log.out']), f"rank {RANK} sees {files}"


def test_log_false_leaves_no_files_on_any_rank(shared_dir):
    _hf(properties=['esp_charges']).get_energy(WATER)
    COMM.Barrier()
    files = os.listdir(shared_dir)
    assert _all_ranks_agree(files == []), f"rank {RANK} sees {files}"


# ---------------------------------------------------------------------------
# errors before QUICK starts stay catchable, and leave no rank behind
# ---------------------------------------------------------------------------

def test_setup_error_raises_on_every_rank_then_next_run_works():
    raised = False
    try:
        _hf().get_energy("Xx 0.0 0.0 0.0")          # unknown element
    except (ValueError, RuntimeError):
        raised = True
    assert _all_ranks_agree(raised), f"rank {RANK}: no error raised"

    result = _hf().get_energy(WATER)
    ok, problems = _identical_to_rank0(result)
    assert ok, f"rank {RANK}: differs from rank 0 in {problems}"
