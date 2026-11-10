"""
pyquick (QUICK's Python interface): smoke test and core tests.

    python3 pyquick_test.py              # smoke test, then the core tests (pytest)
    python3 -m pytest pyquick_test.py    # core tests only

The smoke test is an HF/STO-3G energy of water with the geometry and cutoffs
of ene_H2O_rhf_sto3g.  The core tests cover the energy and its components,
gradients, geometry optimization, a persistent job and .xyz input.  Reference
values come from the quick binary on the same input.  Without pytest installed, only the smoke test runs.
"""

import importlib
import math
import os
import sys

MODULES = {'serial': 'pyquick', 'mpi': 'pyquick.mpi',
           'gpu': 'pyquick.gpu', 'gpu_mpi': 'pyquick.gpu_mpi'}
MODE = os.environ.get('PYQUICK_MODE', 'serial')

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
WATER_XYZ = """3
water, HF/STO-3G reference geometry
O  -0.33840   0.00380   0.23923
H  -0.33510  -0.00190  -0.83277
H   0.67350  -0.00190   0.59353
"""

LOOSE = {'cutoff': '1.0e-9', 'denserms': '1.0e-6'}
TIGHT = {'cutoff': '1.0e-11', 'denserms': '1.0e-8'}
TOL = 4.0e-5

# quick binary references
REF_ENERGY = -74.947863811               # HF/STO-3G water
REF_B3LYP_TOTAL = -75.313101879
REF_B3LYP_EEL = -83.513573608
REF_NUC_REP = 8.200471729
REF_D3_DISP = -0.000024256
REF_D3_TOTAL = -75.313126135
REF_D3BJ_DISP = -0.000575671
REF_D3BJ_TOTAL = -75.313677549
REF_UHF_CATION_TOTAL = -74.667074560     # water cation, charge 1, mult 2
REF_UB3LYP_CATION_TOTAL = -74.975255182
REF_UHF_CATION_GRADIENT = [
    [-0.0326052848,  0.0003655939,  0.0229508980],
    [-0.0001705738, -0.0001825317, -0.0347207565],
    [ 0.0327758586, -0.0001830622,  0.0117698585],
]
REF_TRAJ_DLFIND = [-74.94786381, -74.95030381, -74.96091494, -74.96560980,
                   -74.96588737, -74.96589936, -74.96590121, -74.96590121]
REF_TRAJ_LOPT = [-74.94786211, -74.92888436, -74.91224831, -74.96105792, -74.96414305,
                 -74.96562617, -74.96560561, -74.96585794, -74.96590087, -74.96590110]


def smoke_test(pyquick):
    """HF/STO-3G water energy against the reference; returns the energy."""
    result = pyquick.PyQuick(
        method='HF', basis='STO-3G',
        properties=['mulliken_charges', 'lowdin_charges'],
        keywords={'cutoff': '1.0e-9', 'denserms': '1.0e-6', 'SCF': 300},
    ).get_energy(WATER)

    assert math.isfinite(result.total_energy), \
        f"FAIL  total_energy is not finite: {result.total_energy}"
    diff = abs(result.total_energy - REF_ENERGY)
    assert diff < TOL, (
        f"FAIL  total_energy {result.total_energy:.9f} Ha differs from reference "
        f"{REF_ENERGY:.9f} Ha by {diff:.2e} (tol {TOL:.0e})")
    assert len(result.mulliken) == 3, "FAIL  expected 3 Mulliken charges"
    assert abs(float(sum(result.mulliken))) < 1.0e-3, \
        f"FAIL  Mulliken charges should sum to ~0 for neutral H2O, got {sum(result.mulliken)}"
    return result.total_energy


def main():
    try:
        pyquick = importlib.import_module(MODULES[MODE])
    except (ImportError, KeyError) as e:
        print(f"FAIL  cannot import pyquick (PYQUICK_MODE={MODE}): {e}", file=sys.stderr)
        return 1

    energy = smoke_test(pyquick)
    rank0 = pyquick.mpi_rank() == 0
    if rank0:
        print(f"PASS  smoke test: total_energy = {energy:.9f} Ha  "
              f"(ref {REF_ENERGY:.9f}, diff {abs(energy - REF_ENERGY):.2e})")

    try:
        import pytest
    except ImportError:
        if rank0:
            print("SKIP  pytest not installed: core tests skipped")
        return 0
    sys.stdout.flush()
    args = [__file__, '-q', '-p', 'no:cacheprovider', '--color=no']
    if rank0:
        return int(pytest.main(args))

    # Under MPI every rank runs the tests; only rank 0 prints, unless this rank fails.
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = int(pytest.main(args))
    if code != 0:
        print(f"--- pytest output from MPI rank {pyquick.mpi_rank()} ---\n{buf.getvalue()}")
    return code


if __name__ == '__main__':
    sys.exit(main())


# ---------------------------------------------------------------------------
# core tests (collected by pytest)
# ---------------------------------------------------------------------------

import pytest

try:
    pyquick = importlib.import_module(MODULES[MODE])
except KeyError:
    raise ValueError(f"PYQUICK_MODE={MODE!r}: expected one of {', '.join(MODULES)}")
except ImportError as e:
    pytest.skip(f"pyquick (PYQUICK_MODE={MODE}) is not available: {e}",
                allow_module_level=True)


def _run(method, keywords=LOOSE, geometry=WATER, **kw):
    return pyquick.PyQuick(method=method, basis='STO-3G', keywords=keywords,
                           log=False, **kw).get_energy(geometry)


def _grad():
    return pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False).get_grad(WATER)


def _opt(keywords=LOOSE):
    return pyquick.PyQuick('HF', 'STO-3G', keywords=keywords, log=False).geo_opt(WATER)


# -- energy and its components ------------------------------------------------

def test_energy_matches_reference():
    r = _run('HF')
    assert math.isfinite(r.total_energy)
    assert r.total_energy == pytest.approx(REF_ENERGY, abs=TOL)


def test_energy_decomposition_identities():
    """total = nuclear_repulsion + electronic; electronic = 1e + 2e + xc; xc==0 for HF."""
    r = _run('HF')
    for value in (r.nuclear_repulsion, r.e_electronic, r.e_one_electron,
                  r.e_two_electron, r.e_xc):
        assert math.isfinite(value)
    assert r.total_energy == pytest.approx(r.nuclear_repulsion + r.e_electronic, abs=1e-8)
    assert r.e_electronic == pytest.approx(
        r.e_one_electron + r.e_two_electron + r.e_xc, abs=1e-8)
    assert r.e_xc == pytest.approx(0.0, abs=1e-10)


def test_component_total_energy_dft():
    assert _run('B3LYP').total_energy == pytest.approx(REF_B3LYP_TOTAL, abs=TOL)


def test_component_e_electronic_dft():
    assert _run('B3LYP').e_electronic == pytest.approx(REF_B3LYP_EEL, abs=TOL)


def test_component_nuclear_repulsion():
    """Same geometry -> same nuclear repulsion for every method."""
    for method in ('HF', 'B3LYP'):
        assert _run(method).nuclear_repulsion == pytest.approx(REF_NUC_REP, abs=1e-8)


def test_component_e_xc_zero_for_hf_after_dft():
    """Hartree-Fock has no XC term, even right after a DFT run."""
    _run('B3LYP')
    r = _run('HF')
    assert r.e_xc == 0.0
    assert r.e_two_electron == pytest.approx(r.e_electronic - r.e_one_electron, abs=1e-10)


def test_components_sum_to_total():
    """total = e_electronic + nuclear_repulsion (+ e_dispersion when set);
    e_electronic = e_one_electron + e_two_electron + e_xc."""
    for r in (_run('HF'), _run('B3LYP'), _run('B3LYP', keywords={**LOOSE, 'D3': None})):
        disp = r.e_dispersion if 'e_dispersion' in r._values else 0.0
        assert r.total_energy == pytest.approx(
            r.e_electronic + r.nuclear_repulsion + disp, abs=1e-10)
        assert r.e_electronic == pytest.approx(
            r.e_one_electron + r.e_two_electron + r.e_xc, abs=1e-10)


@pytest.mark.parametrize("method, kw", [('HF', {}), ('B3LYP', {}),
                                        ('B3LYP', {'charge': 1, 'mult': 2})])
def test_kinetic_and_attraction_add_up_to_one_electron_energy(method, kw):
    r = _run(method, **kw)
    assert r.e_kinetic + r.e_nuclear_attraction == pytest.approx(r.e_one_electron, abs=1e-10)
    assert r.e_kinetic > 0 and r.e_nuclear_attraction < 0


@pytest.mark.parametrize("keyword, ref_disp, ref_total", [
    ('D3', REF_D3_DISP, REF_D3_TOTAL),
    ('D3BJ', REF_D3BJ_DISP, REF_D3BJ_TOTAL),
])
def test_dispersion_matches_binary(keyword, ref_disp, ref_total):
    r = _run('B3LYP', keywords={**LOOSE, keyword: None})
    assert r.e_dispersion == pytest.approx(ref_disp, abs=1e-9)
    assert r.total_energy == pytest.approx(ref_total, abs=TOL)


def test_open_shell_uhf_energy_matches_binary():
    r = _run('HF', charge=1, mult=2)
    assert r.total_energy == pytest.approx(REF_UHF_CATION_TOTAL, abs=TOL)
    assert r.metadata['calc_keyword'] == 'UHF'


def test_open_shell_udft_energy_matches_binary():
    r = _run('B3LYP', charge=1, mult=2)
    assert r.total_energy == pytest.approx(REF_UB3LYP_CATION_TOTAL, abs=TOL)
    assert r.metadata['calc_keyword'] == 'UDFT'


# -- gradient -----------------------------------------------------------------

def test_get_grad_returns_energy_and_gradient():
    r = _grad()
    # energy is produced by the same job (no separate energy calculation)
    assert r.total_energy == pytest.approx(REF_ENERGY, abs=TOL)
    g = r.gradient
    assert g.shape == (3, 3)                  # (natom, 3), Hartree/Bohr
    assert all(math.isfinite(v) for v in g.ravel())
    # translational invariance: the net force on the molecule is ~0
    assert max(abs(g.sum(axis=0))) < 1.0e-4


def test_gradient_matches_finite_difference():
    """Central finite differences of the energy reproduce the analytic gradient."""
    ANG_PER_BOHR = 0.52917721067
    h = 0.005  # Angstrom
    atoms = [ln.split() for ln in WATER.strip().splitlines()]

    def energy_at(atom, comp, delta):
        moved = [row[:] for row in atoms]
        moved[atom][1 + comp] = f"{float(moved[atom][1 + comp]) + delta:.6f}"
        geom = "\n".join(" ".join(row) for row in moved)
        return _run('HF', keywords=TIGHT, geometry=geom).total_energy

    g = _grad().gradient          # Hartree/Bohr
    for atom in (0, 1):
        for comp in (0, 1, 2):
            fd = (energy_at(atom, comp, +h) - energy_at(atom, comp, -h)) / (2.0 * h / ANG_PER_BOHR)
            assert g[atom, comp] == pytest.approx(fd, abs=2.0e-3)


def test_open_shell_gradient_matches_binary():
    r = pyquick.PyQuick('HF', 'STO-3G', charge=1, mult=2, keywords=TIGHT,
                        log=False).get_grad(WATER)
    for got, ref in zip(r.gradient, REF_UHF_CATION_GRADIENT):
        assert list(got) == pytest.approx(ref, abs=1e-6)


# -- geometry optimization ----------------------------------------------------

def test_geo_opt_optimizes_water():
    r = _opt(keywords=TIGHT)
    assert r.converged
    assert r.optimized_coordinates.shape == (3, 3)
    assert math.isfinite(r.total_energy)
    # the optimized structure must not be higher in energy than the input one
    assert r.total_energy <= REF_ENERGY + 1e-6
    # `coordinates` still holds the input geometry, not the optimized one
    assert r.coordinates[0] == pytest.approx([-0.33840, 0.00380, 0.23923], abs=1e-4)


def test_trajectory_dlfind_matches_binary():
    assert list(_opt().trajectory_energies) == pytest.approx(REF_TRAJ_DLFIND, abs=1e-6)


def test_trajectory_lopt_matches_binary():
    r = _opt(keywords={**LOOSE, 'LOPT': None})
    assert list(r.trajectory_energies) == pytest.approx(REF_TRAJ_LOPT, abs=1e-6)


# -- persistent job and .xyz input --------------------------------------------

def test_persistent_dft_job_with_moving_geometry():
    calc = pyquick.PyQuick('B3LYP', 'STO-3G', keywords=LOOSE, log=False)
    geometries = [WATER, WATER_STRETCHED, WATER, WATER_STRETCHED]
    with calc.new_job(WATER) as job:
        steps = [job.run(g, gradient=(k == 3)) for k, g in enumerate(geometries)]
    # each step matches a one-shot run at the same geometry
    ref = {g: calc.get_energy(g).total_energy for g in (WATER, WATER_STRETCHED)}
    for g, r in zip(geometries, steps):
        assert r.total_energy == pytest.approx(ref[g], abs=TOL)
    assert steps[3].gradient.shape == (3, 3)


def test_get_energy_from_xyz_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mol.xyz").write_text(WATER_XYZ)
    assert _run('HF', geometry='mol.xyz').total_energy == pytest.approx(REF_ENERGY, abs=TOL)
