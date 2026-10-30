"""
GPU-MPI tests for pyquick: one GPU per MPI rank (cmake -DCUDA=TRUE -DMPI=TRUE
-DPYTHON=TRUE, bound to libquick_mpi_cuda).  Run INSIDE mpiexec, with at least
as many visible GPUs as ranks:

    CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python -m mpi4py -m pytest test_pyquick_gpu_mpi.py -q

Every rank runs the whole file.  The module is skipped on any other build
(serial, CPU MPI, single GPU).  test_pyquick_mpi.py and test_pyquick_api.py also
run unchanged on this build; this file adds what only the GPU-MPI build can get
wrong: that every rank really drives its own GPU, and that results match the
quick-binary references and agree bit for bit across ranks.

Same rule as test_pyquick_mpi.py: every check is reduced across ranks before it
can fail, so a failure fails on every rank instead of leaving the others
waiting in a later collective.
"""

import os
import re
import shutil
import tempfile

import numpy as np

import pytest

pyquick = pytest.importorskip("pyquick")

if not (getattr(pyquick, "gpu_enabled", lambda: False)()
        and getattr(pyquick, "mpi_enabled", lambda: False)()):
    pytest.skip("pyquick was not built for GPU + MPI (-DCUDA=TRUE -DMPI=TRUE)",
                allow_module_level=True)

from mpi4py import MPI  # noqa: E402

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
LOOSE = {'cutoff': '1.0e-9', 'denserms': '1.0e-6'}
TOL = 4.0e-5

# quick-binary references, same inputs as test_pyquick_api.py
REF_HF_ENERGY = -74.947863811        # HF/STO-3G water, LOOSE
REF_B3LYP_ENERGY = -75.313101879     # B3LYP/STO-3G water, LOOSE

RESULT_FIELDS = (
    'total_energy', 'nuclear_repulsion', 'e_electronic', 'e_one_electron',
    'e_two_electron', 'e_xc', 'e_dispersion', 'e_external_charge',
    'atomic_numbers', 'coordinates', 'mo_energies', 'density_matrix',
    'beta_mo_energies', 'beta_density_matrix', 'mulliken', 'lowdin', 'dipole',
    'esp_charges', 'gradient', 'optimized_coordinates', 'converged',
)


# ---------------------------------------------------------------------------
# helpers (as in test_pyquick_mpi.py)
# ---------------------------------------------------------------------------

def _all_ranks_agree(ok):
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
    """One directory every rank works in (pytest's tmp_path differs per rank)."""
    path = tempfile.mkdtemp(prefix="pyquick_gpu_mpi_") if RANK == 0 else None
    path = COMM.bcast(path, root=0)
    monkeypatch.chdir(path)
    yield path
    COMM.Barrier()
    if RANK == 0:
        shutil.rmtree(path, ignore_errors=True)


def _hf(**kw):
    return pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False, **kw)


def _b3lyp(**kw):
    return pyquick.PyQuick('B3LYP', 'STO-3G', keywords=LOOSE, log=False, **kw)


# ---------------------------------------------------------------------------
# the build, and one GPU per rank
# ---------------------------------------------------------------------------

def test_build_is_gpu_mpi():
    ok = (pyquick.gpu_enabled() and pyquick.mpi_enabled()
          and pyquick.mpi_size() == SIZE and pyquick.mpi_rank() == RANK)
    assert _all_ranks_agree(ok), (
        f"rank {RANK}: gpu={pyquick.gpu_enabled()} mpi={pyquick.mpi_enabled()} "
        f"rank/size={pyquick.mpi_rank()}/{pyquick.mpi_size()}")


def _rank_devices_in_log(path):
    """[(mpi rank, device id), ...] from QUICK's multi-GPU log block:
         |        --    MPI RANK   0 --
         |   CUDA DEVICE IN USE          :        0
    """
    pairs, rank = [], None
    with open(path) as f:
        for line in f:
            m = re.search(r'MPI RANK\s+(\d+)', line)
            if m:
                rank = int(m.group(1))
                continue
            m = re.search(r'DEVICE IN USE\s*:\s*(\d+)', line)
            if m and rank is not None:
                pairs.append((rank, int(m.group(1))))
                rank = None
    return pairs


def test_every_rank_has_its_own_gpu(shared_dir):
    pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE).get_energy(WATER, name='gpu_ranks')
    COMM.Barrier()
    pairs = _rank_devices_in_log('gpu_ranks.out') if RANK == 0 else None
    pairs = COMM.bcast(pairs, root=0)
    ranks = sorted(r for r, _ in pairs)
    devices = [d for _, d in pairs]
    ok = ranks == list(range(SIZE)) and len(set(devices)) == SIZE
    assert ok, f"log lists (rank, device) = {pairs}; expected {SIZE} ranks on {SIZE} distinct GPUs"


# ---------------------------------------------------------------------------
# results: match the quick binary, identical on every rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,ref", [('HF', REF_HF_ENERGY), ('B3LYP', REF_B3LYP_ENERGY)])
def test_energy_matches_binary_reference(method, ref):
    r = pyquick.PyQuick(method, 'STO-3G', keywords=LOOSE, log=False).get_energy(WATER)
    assert _all_ranks_agree(abs(r.total_energy - ref) < TOL), (
        f"rank {RANK}: {method} E = {r.total_energy:.9f}, reference {ref:.9f}")


CASES = {
    'energy':       lambda: _hf().get_energy(WATER),
    'grad':         lambda: _hf().get_grad(WATER),
    'b3lyp':        lambda: _b3lyp().get_energy(WATER),
    'b3lyp_grad':   lambda: _b3lyp().get_grad(WATER),
    'uhf':          lambda: _hf(charge=1, mult=2).get_energy(WATER),
    'geo_opt':      lambda: _hf().geo_opt(WATER),
    'props':        lambda: _hf(properties=['mulliken_charges', 'dipole',
                                            'esp_charges']).get_energy(WATER),
}


@pytest.mark.parametrize("case", list(CASES))
def test_identical_results_on_all_ranks(case):
    ok, problems = _identical_to_rank0(CASES[case]())
    assert ok, f"rank {RANK}: differs from rank 0 in {problems}"


@pytest.mark.parametrize("method", ['HF', 'B3LYP'])
def test_persistent_job_on_gpus(method):
    calc = pyquick.PyQuick(method, 'STO-3G', keywords=LOOSE, log=False)
    geometries = [WATER, WATER_STRETCHED, WATER]
    with calc.new_job(WATER) as job:
        steps = [job.run(g, gradient=(k == 2)) for k, g in enumerate(geometries)]
    for k, r in enumerate(steps):
        ok, problems = _identical_to_rank0(r)
        assert ok, f"rank {RANK}, step {k}: differs from rank 0 in {problems}"
    assert steps[0].total_energy == pytest.approx(steps[2].total_energy, abs=TOL)
