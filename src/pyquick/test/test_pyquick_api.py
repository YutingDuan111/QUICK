"""
pytest unit + integration tests for the pyquick PyQuick/Result API.

Pure-Python tests (validation, method resolution, input assembly) need only that
the compiled extension imports.  The integration tests additionally run a small
HF/STO-3G water job.

Job type is chosen by which method you call:
    calc.get_energy(geometry)  -> single-point energy
    calc.get_grad(geometry)    -> energy + forces
    calc.geo_opt(geometry)     -> optimized geometry
`properties` selects only optional extras (charges, MO energies, density matrix).
"""

import math
import os

import numpy as np

import pytest

pyquick = pytest.importorskip("pyquick")


WATER = """
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
"""
REF_ENERGY = -74.947863811
TOL = 4.0e-5


# ---------------------------------------------------------------------------
# Pure-Python: construction & validation
# ---------------------------------------------------------------------------

def test_missing_basis_raises():
    with pytest.raises(ValueError):
        pyquick.PyQuick(method='HF', basis='')


def test_unknown_property_raises():
    with pytest.raises(ValueError):
        pyquick.PyQuick(method='HF', basis='STO-3G', properties=['bogus'])


def test_properties_default_empty():
    """No properties needed for an energy run; energy is always returned."""
    calc = pyquick.PyQuick(method='HF', basis='STO-3G')
    assert calc.properties == set()


def test_jobtype_as_property_is_rejected():
    """energy/gradient/optimize are methods now, so they are not valid property values."""
    for bad in ('energy', 'gradient', 'forces', 'optimize', 'optimized_coords'):
        with pytest.raises(ValueError):
            pyquick.PyQuick(method='HF', basis='STO-3G', properties=[bad])


@pytest.mark.parametrize("prop", ['mp2', 'e_mp2', 'MP2'])
def test_mp2_property_not_supported(prop):
    """MP2 is deliberately not part of pyquick: a plain 'not supported' error,
    with no promise of a later phase."""
    with pytest.raises(ValueError) as excinfo:
        pyquick.PyQuick(method='HF', basis='STO-3G', properties=[prop])
    message = str(excinfo.value)
    assert 'not supported' in message
    assert 'Phase' not in message and 'planned' not in message


def test_dipole_is_a_supported_property():
    calc = pyquick.PyQuick(method='HF', basis='STO-3G', properties=['dipole'])
    assert 'dipole' in calc.properties


@pytest.mark.parametrize("method,mult,expected", [
    ('HF',    1, ('HF', None)),
    ('RHF',   1, ('HF', None)),
    ('HF',    2, ('UHF', None)),
    ('UHF',   1, ('UHF', None)),
    ('B3LYP', 1, ('DFT', 'B3LYP')),
    ('B3LYP', 2, ('UDFT', 'B3LYP')),
    ('pbe0',  1, ('DFT', 'PBE0')),
])
def test_resolve_method(method, mult, expected):
    assert pyquick._resolve_method(method, mult) == expected


def test_resolve_method_dft_without_functional():
    with pytest.raises(ValueError):
        pyquick._resolve_method('DFT', 1)


def test_resolve_method_bad_mult():
    with pytest.raises(ValueError):
        pyquick._resolve_method('HF', 0)


def test_input_string_contains_tokens():
    calc = pyquick.PyQuick(
        method='B3LYP', basis='6-31G*',
        properties=['mulliken_charges'],
        charge=-1, mult=2, keywords={'cutoff': '1e-9'})
    s = calc.input_string.upper()
    assert 'UDFT' in s          # functional + open shell -> UDFT
    assert 'B3LYP' in s
    assert 'DIPOLE' in s        # charges -> DIPOLE keyword
    assert 'CHARGE=-1' in s
    assert 'MULT=2' in s
    assert 'CUTOFF=1E-9' in s
    assert 'BASIS=6-31G*' in s


# ---------------------------------------------------------------------------
# geo_opt: DL-Find (default) optimization
# ---------------------------------------------------------------------------

def test_geo_opt_optimizes_water():
    calc = pyquick.PyQuick(
        method='HF', basis='STO-3G',
        keywords={'cutoff': '1.0e-11', 'denserms': '1.0e-8'})
    r = calc.geo_opt(WATER)
    assert r.converged
    assert r.optimized_coordinates.shape == (3, 3)
    assert math.isfinite(r.total_energy)
    # the optimized structure must not be higher in energy than the input one
    assert r.total_energy <= REF_ENERGY + 1e-6
    # `coordinates` still holds the input geometry, not the optimized one
    assert r.coordinates[0] == pytest.approx([-0.33840, 0.00380, 0.23923], abs=1e-4)


# ---------------------------------------------------------------------------
# Integration: get_energy() requires a working SCF run
# ---------------------------------------------------------------------------

def _energy(properties=()):
    calc = pyquick.PyQuick(
        method='HF', basis='STO-3G', properties=properties,
        keywords={'cutoff': '1.0e-9', 'denserms': '1.0e-6'})
    return calc.get_energy(WATER)


def test_energy_matches_reference():
    result = _energy()
    assert math.isfinite(result.total_energy)
    assert abs(result.total_energy - REF_ENERGY) < TOL


def test_energy_decomposition_identities():
    """total = nuclear_repulsion + electronic; electronic = 1e + 2e + xc; xc==0 for HF."""
    r = _energy()
    for value in (r.nuclear_repulsion, r.e_electronic, r.e_one_electron,
                  r.e_two_electron, r.e_xc):
        assert math.isfinite(value)
    assert r.total_energy == pytest.approx(r.nuclear_repulsion + r.e_electronic, abs=1e-8)
    assert r.e_electronic == pytest.approx(
        r.e_one_electron + r.e_two_electron + r.e_xc, abs=1e-8)
    assert r.e_xc == pytest.approx(0.0, abs=1e-10)   # pure HF


def test_conditional_energies_gated():
    """dispersion only present if requested; external charge only if point_charges given.

    The AttributeError must explain *why* the value is unavailable and how to enable it.
    """
    r = _energy()   # no D* keyword, no point_charges
    with pytest.raises(AttributeError, match="(?i)dispersion"):
        _ = r.e_dispersion
    with pytest.raises(AttributeError, match="point_charges"):
        _ = r.e_external_charge


def test_unrequested_property_raises_attributeerror():
    """Not-requested opt-in properties raise AttributeError naming the property to request.

    density_matrix (and mo_energies) are NOT gated this way: they're always-on
    extras every SCF computes for free, so they're available even when not named
    in `properties` (see _ALWAYS_ON_PROPERTIES in __init__.py).
    """
    result = _energy()
    with pytest.raises(AttributeError, match="mulliken_charges"):
        _ = result.mulliken
    assert result.density_matrix.ndim == 2


def test_requested_charges_available():
    result = _energy(['mulliken_charges', 'lowdin_charges'])
    assert len(result.mulliken) == 3
    assert len(result.lowdin) == 3
    assert abs(float(sum(result.mulliken))) < 1.0e-3


def test_density_matrix_shape():
    result = _energy(['density_matrix'])
    dm = result.density_matrix
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1]


def test_geometry_available_from_quick_parse():
    """Geometry (atomic numbers + Angstrom coords) is always exposed, from QUICK's parse."""
    result = _energy()
    assert result.atomic_numbers.tolist() == [8, 1, 1]
    assert result.coordinates.shape == (3, 3)
    # coordinates round-trip the input geometry (Angstrom)
    assert result.coordinates[0] == pytest.approx([-0.33840, 0.00380, 0.23923], abs=1e-4)


# ---------------------------------------------------------------------------
# 2.0 Dipole moment
# ---------------------------------------------------------------------------

def test_dipole_available_when_requested():
    result = _energy(['dipole'])
    d = result.dipole
    assert d.shape == (3,)
    magnitude = math.sqrt(float(sum(x * x for x in d)))   # Debye
    assert 0.3 < magnitude < 4.0            # water is polar (~1.7 D at HF/STO-3G)


def test_dipole_not_requested_raises():
    result = _energy()                       # dipole not in properties
    with pytest.raises(AttributeError, match="dipole"):
        _ = result.dipole


# ---------------------------------------------------------------------------
# 2.1 Nuclear gradient (get_grad) — energy + gradient computed together
# ---------------------------------------------------------------------------

def _grad(properties=()):
    calc = pyquick.PyQuick(
        method='HF', basis='STO-3G', properties=properties,
        keywords={'cutoff': '1.0e-11', 'denserms': '1.0e-8'})
    return calc.get_grad(WATER)


def test_get_grad_returns_energy_and_gradient():
    r = _grad()
    # energy is produced by the same job (no separate energy calculation)
    assert abs(r.total_energy - REF_ENERGY) < TOL
    g = r.gradient
    assert g.shape == (3, 3)                  # (natom, 3), Hartree/Bohr
    assert all(math.isfinite(v) for v in g.ravel())
    # translational invariance: the net force on the molecule is ~0
    net = g.sum(axis=0)
    assert max(abs(net)) < 1.0e-4


def test_gradient_unavailable_from_get_energy():
    r = _energy()
    with pytest.raises(AttributeError, match="get_grad"):
        _ = r.gradient


def test_gradient_matches_finite_difference():
    """Central finite differences of the energy reproduce the analytic gradient."""
    ANG_PER_BOHR = 0.52917721067
    h = 0.005  # Angstrom
    atoms = [ln.split() for ln in WATER.strip().splitlines()]

    def energy_at(atom, comp, delta):
        moved = [row[:] for row in atoms]
        moved[atom][1 + comp] = f"{float(moved[atom][1 + comp]) + delta:.6f}"
        geom = "\n".join(" ".join(row) for row in moved)
        calc = pyquick.PyQuick(
            method='HF', basis='STO-3G',
            keywords={'cutoff': '1.0e-11', 'denserms': '1.0e-8'})
        return calc.get_energy(geom).total_energy

    g = _grad().gradient          # Hartree/Bohr
    for atom in (0, 1):
        for comp in (0, 1, 2):
            e_plus = energy_at(atom, comp, +h)
            e_minus = energy_at(atom, comp, -h)
            # dE/dx in Hartree/Bohr:  (Ang step) -> (Bohr step)
            fd = (e_plus - e_minus) / (2.0 * h / ANG_PER_BOHR)
            assert g[atom, comp] == pytest.approx(fd, abs=2.0e-3)


# ---------------------------------------------------------------------------
# copy() / deepcopy(): mirror Python's copy.copy / copy.deepcopy
#
# Shallow copy shares mutable members (keywords dict, properties set, result
# arrays); deep copy makes them independent.  In both cases a PyQuick copy gets
# its own engine -- the engine is runtime state, never shared between objects.
# ---------------------------------------------------------------------------

import copy as _copy


def _calc():
    return pyquick.PyQuick(
        method='HF', basis='STO-3G', properties=['mulliken_charges'],
        keywords={'cutoff': '1.0e-9', 'denserms': '1.0e-6'}, log=False)


def test_pyquick_copy_is_shallow():
    a = _calc()
    b = a.copy()
    assert b is not a and isinstance(b, pyquick.PyQuick)
    assert (b.method, b.basis, b.charge, b.mult, b.log) == \
           (a.method, a.basis, a.charge, a.mult, a.log)
    # mutable members are shared, like copy.copy
    assert b.keywords is a.keywords
    assert b.properties is a.properties


def test_pyquick_deepcopy_is_independent():
    a = _calc()
    b = a.deepcopy()
    assert b.keywords == a.keywords and b.keywords is not a.keywords
    assert b.properties == a.properties and b.properties is not a.properties
    b.keywords['scf'] = '50'
    b.properties.add('dipole')
    assert 'scf' not in a.keywords
    assert 'dipole' not in a.properties


def test_pyquick_copy_module_dispatches_to_methods():
    a = _calc()
    shallow = _copy.copy(a)
    deep = _copy.deepcopy(a)
    assert shallow.keywords is a.keywords
    assert deep.keywords is not a.keywords and deep.keywords == a.keywords


@pytest.mark.parametrize("how", ["copy", "deepcopy"])
def test_pyquick_copy_does_not_share_engine(how):
    """A copy made after a run must build its own engine, so edits to the
    copy take effect instead of reusing the source's stale engine."""
    a = _calc()
    a.get_energy(WATER)
    b = getattr(a, how)()
    assert b._engine is None or b._engine is not a._engine
    b.basis = '6-31G'
    assert 'BASIS=6-31G' in b.input_string.upper()
    assert 'BASIS=STO-3G' in a.input_string.upper()


@pytest.mark.parametrize("how", ["copy", "deepcopy"])
def test_pyquick_copy_runs_same_energy(how):
    a = _calc()
    ra = a.get_energy(WATER)
    rb = getattr(a, how)().get_energy(WATER)
    assert rb.total_energy == pytest.approx(ra.total_energy, abs=1e-10)


def test_result_copy_is_shallow():
    r = _calc().get_energy(WATER)
    c = r.copy()
    assert c is not r and isinstance(c, pyquick.Result)
    assert c.total_energy == r.total_energy
    # arrays are shared, like copy.copy
    assert c.density_matrix is r.density_matrix
    assert c.mulliken is r.mulliken
    assert _copy.copy(r).density_matrix is r.density_matrix


def test_result_deepcopy_is_independent():
    r = _calc().get_energy(WATER)
    d = r.deepcopy()
    assert d.total_energy == r.total_energy
    assert d.density_matrix is not r.density_matrix
    assert (d.density_matrix == r.density_matrix).all()
    d.mulliken[0] = 99.0
    assert r.mulliken[0] != 99.0
    assert d.metadata == r.metadata and d.metadata is not r.metadata
    assert _copy.deepcopy(r).density_matrix is not r.density_matrix


def test_result_copy_keeps_property_gating():
    """A copied Result still raises for properties that were not requested."""
    r = _calc().get_energy(WATER)
    for c in (r.copy(), r.deepcopy()):
        with pytest.raises(AttributeError, match="lowdin_charges"):
            _ = c.lowdin


def test_engine_deepcopy_starts_unrun():
    """copy.deepcopy on the private engine must not clone a 'ran' engine
    (its __del__ would then finalize QUICK state it does not own)."""
    a = _calc()
    a.get_energy(WATER)
    e = _copy.deepcopy(a._engine)
    assert e._ran is False
    assert e._results == {}


@pytest.mark.parametrize("fn", [_copy.copy, _copy.deepcopy])
def test_job_cannot_be_copied(fn):
    """A Job owns live Fortran allocations; copying it is refused."""
    with _calc().new_job(WATER) as job:
        with pytest.raises(TypeError):
            fn(job)


# ---------------------------------------------------------------------------
# point_charges=: external point charges, passed per call
#
# Each charge is (x, y, z, q) with x, y, z in Angstrom.  The charges act on the
# molecule (SCF density, nuclear repulsion, gradient).  Their mutual repulsion
# ECharge is reported as e_external_charge but is NEVER part of total_energy,
# matching the quick binary.  References below come from the quick binary on
# the same inputs (water + two +1 charges 1 Angstrom apart).
# ---------------------------------------------------------------------------

CHARGES = [(5.0, 0.0, 0.0, 1.0), (6.0, 0.0, 0.0, 1.0)]
REF_PC_ENERGY = -74.939356355     # quick binary, EXTCHARGES, cutoff 1e-9
REF_PC_ECHARGE = 0.529177209      # 1 / (1 Angstrom in bohr)
REF_PC_OPT_ENERGY = -74.956315292 # quick binary, EXTCHARGES OPTIMIZE
REF_PC_GRADIENT = [               # quick binary, EXTCHARGES GRADIENT, Hartree/Bohr
    [-0.0402633390,  0.0004027813,  0.0303094245],
    [-0.0231180315, -0.0002168710, -0.0797239350],
    [ 0.0648798397, -0.0001766444,  0.0497540855],
]
ANG_TO_BOHR = 1.0 / 0.52917721067


def _pc_calc(tight=False):
    kw = ({'cutoff': '1.0e-11', 'denserms': '1.0e-8'} if tight
          else {'cutoff': '1.0e-9', 'denserms': '1.0e-6'})
    return pyquick.PyQuick(method='HF', basis='STO-3G', keywords=kw, log=False)


def test_point_charges_energy_matches_binary():
    r = _pc_calc().get_energy(WATER, point_charges=CHARGES)
    assert r.total_energy == pytest.approx(REF_PC_ENERGY, abs=TOL)
    assert r.e_external_charge == pytest.approx(REF_PC_ECHARGE, abs=1e-8)


def test_point_charges_echarge_not_in_total_energy():
    """total_energy = e_electronic + nuclear_repulsion; ECharge is reported only."""
    r = _pc_calc().get_energy(WATER, point_charges=CHARGES)
    assert r.total_energy == pytest.approx(
        r.e_electronic + r.nuclear_repulsion, abs=1e-10)
    assert r.e_external_charge > 0.5      # large, so leaving it out is a real check


def test_point_charges_add_atom_charge_repulsion():
    """nuclear_repulsion gains exactly sum_i sum_j Z_i q_j / r_ij (bohr)."""
    calc = _pc_calc()
    vac = calc.get_energy(WATER)
    pc = calc.get_energy(WATER, point_charges=CHARGES)
    atoms = [(ln.split()[0], [float(v) for v in ln.split()[1:4]])
             for ln in WATER.strip().splitlines()]
    z = {'O': 8.0, 'H': 1.0}
    expected = 0.0
    for sym, a in atoms:
        for *c, q in CHARGES:
            r_bohr = math.dist(a, c) * ANG_TO_BOHR
            expected += z[sym] * q / r_bohr
    assert pc.nuclear_repulsion - vac.nuclear_repulsion == pytest.approx(expected, abs=1e-8)


def test_point_charges_gradient_matches_binary():
    r = _pc_calc(tight=True).get_grad(WATER, point_charges=CHARGES)
    assert r.e_external_charge == pytest.approx(REF_PC_ECHARGE, abs=1e-8)
    for got, ref in zip(r.gradient, REF_PC_GRADIENT):
        assert list(got) == pytest.approx(ref, abs=1e-6)
    # the charges exert a net force, unlike the isolated molecule
    assert abs(r.gradient.sum(axis=0)[0]) > 1e-3


def test_point_charges_geo_opt_matches_binary():
    r = _pc_calc(tight=True).geo_opt(WATER, point_charges=CHARGES)
    assert r.converged
    assert r.total_energy == pytest.approx(REF_PC_OPT_ENERGY, abs=TOL)
    assert r.e_external_charge == pytest.approx(REF_PC_ECHARGE, abs=1e-8)


def test_point_charges_do_not_leak_into_next_call():
    """Charges belong to one call; the next call without them is in vacuum."""
    calc = _pc_calc()
    with_pc = calc.get_energy(WATER, point_charges=CHARGES)
    vac = calc.get_energy(WATER)
    assert with_pc.total_energy == pytest.approx(REF_PC_ENERGY, abs=TOL)
    assert vac.total_energy == pytest.approx(REF_ENERGY, abs=TOL)
    with pytest.raises(AttributeError, match="point_charges"):
        _ = vac.e_external_charge


def test_point_charges_accepts_numpy_array():
    import numpy as np
    r = _pc_calc().get_energy(WATER, point_charges=np.array(CHARGES))
    assert r.total_energy == pytest.approx(REF_PC_ENERGY, abs=TOL)


def test_single_point_charge_has_zero_echarge():
    """One charge has no partner: e_external_charge is available and 0."""
    r = _pc_calc().get_energy(WATER, point_charges=[CHARGES[0]])
    assert r.e_external_charge == 0.0
    assert r.total_energy != pytest.approx(REF_ENERGY, abs=1e-4)   # still felt


@pytest.mark.parametrize("bad", [
    [(5.0, 0.0, 0.0)],                 # missing charge
    [(5.0, 0.0, 0.0, 1.0, 2.0)],       # extra column
    [],                                # empty
    [('a', 0.0, 0.0, 1.0)],            # not numeric
])
def test_point_charges_malformed_raises(bad):
    with pytest.raises(ValueError, match="point_charges"):
        _pc_calc().get_energy(WATER, point_charges=bad)


def test_extcharges_keyword_rejected():
    """The bare EXTCHARGES keyword used to run silently in vacuum; now it
    points the user to point_charges=."""
    with pytest.raises(ValueError, match="point_charges"):
        pyquick.PyQuick(method='HF', basis='STO-3G', keywords={'EXTCHARGES': None})
    with pytest.raises(ValueError, match="point_charges"):
        pyquick.PyQuick(method='HF', basis='STO-3G', keywords={'extcharges': None})


def test_persistent_job_with_point_charges():
    """new_job(point_charges=) fixes the charges for every step of the job."""
    calc = _pc_calc()
    with calc.new_job(WATER, point_charges=CHARGES) as job:
        r1 = job.run(WATER)
        r2 = job.run(WATER)
    for r in (r1, r2):
        assert r.total_energy == pytest.approx(REF_PC_ENERGY, abs=TOL)
        assert r.e_external_charge == pytest.approx(REF_PC_ECHARGE, abs=1e-8)


# ---------------------------------------------------------------------------
# Energy components, dispersion, open shell, ESP
#
# Unless noted, references come from the quick binary on the same water
# geometry and keywords (one calculation per process, so no carried-over state).
# The binary does not print E1e / E2e / Exc, so those are checked with exact
# physical identities instead (a one-electron system has no two-electron
# energy; Hartree-Fock has no XC energy).
# ---------------------------------------------------------------------------

LOOSE = {'cutoff': '1.0e-9', 'denserms': '1.0e-6'}
TIGHT = {'cutoff': '1.0e-11', 'denserms': '1.0e-8'}
H2 = "H 0.0 0.0 0.0\nH 0.0 0.0 0.74"

REF_B3LYP_TOTAL = -75.313101879
REF_B3LYP_EEL = -83.513573608
REF_NUC_REP = 8.200471729            # same water geometry for every method
REF_D3_DISP = -0.000024256
REF_D3_TOTAL = -75.313126135
REF_D3BJ_DISP = -0.000575671
REF_D3BJ_TOTAL = -75.313677549
REF_D3_GRADIENT = [                  # B3LYP D3, tight cutoffs, Hartree/Bohr
    [-0.0005385806, -0.0000097318, -0.0001152833],
    [-0.0276453328, -0.0000015800, -0.0392502864],
    [ 0.0281839134,  0.0000113118,  0.0393655698],
]
REF_UHF_CATION_TOTAL = -74.667074560     # H2O+, charge=1 mult=2
REF_UB3LYP_CATION_TOTAL = -74.975255182
REF_UHF_CATION_OPT_TOTAL = -74.669743222
REF_UHF_CATION_MULLIKEN = [0.1335, 0.4333, 0.4333]   # printed to 4 decimals
REF_UHF_CATION_LOWDIN = [0.2876, 0.3562, 0.3562]
REF_UHF_CATION_DIPOLE = [2.2675, -0.0255, -1.6030]   # Debye
REF_UHF_CATION_GRADIENT = [
    [-0.0326052848,  0.0003655939,  0.0229508980],
    [-0.0001705738, -0.0001825317, -0.0347207565],
    [ 0.0327758586, -0.0001830622,  0.0117698585],
]


def _run(method, keywords=LOOSE, geometry=WATER, **kw):
    return pyquick.PyQuick(method=method, basis='STO-3G', keywords=keywords,
                           log=False, **kw).get_energy(geometry)


# -- each energy component ----------------------------------------------------

def test_component_total_energy_dft():
    assert _run('B3LYP').total_energy == pytest.approx(REF_B3LYP_TOTAL, abs=TOL)


def test_component_e_electronic_dft():
    assert _run('B3LYP').e_electronic == pytest.approx(REF_B3LYP_EEL, abs=TOL)


def test_component_nuclear_repulsion():
    """Same geometry -> same nuclear repulsion for every method."""
    for method in ('HF', 'B3LYP'):
        assert _run(method).nuclear_repulsion == pytest.approx(REF_NUC_REP, abs=1e-8)


def test_component_e_one_electron_one_electron_system():
    """H2+ has a single electron, so all of its electronic energy is one-electron."""
    r = _run('HF', geometry=H2, charge=1, mult=2)
    assert r.e_one_electron == pytest.approx(r.e_electronic, abs=1e-8)


def test_component_e_two_electron_zero_for_one_electron_after_dft():
    """H2+ has no electron pairs: e_two_electron must be 0, even right after a
    DFT run in the same process (QUICK only sets Exc on the DFT path)."""
    _run('B3LYP', geometry=H2, charge=1, mult=2)
    r = _run('HF', geometry=H2, charge=1, mult=2)
    assert r.e_two_electron == pytest.approx(0.0, abs=1e-8)


def test_component_e_xc_zero_for_hf_after_dft():
    """Hartree-Fock has no XC term, even right after a DFT run."""
    _run('B3LYP')
    r = _run('HF')
    assert r.e_xc == 0.0
    assert r.e_two_electron == pytest.approx(r.e_electronic - r.e_one_electron, abs=1e-10)


def test_component_e_xc_dft_independent_of_previous_run():
    fresh = _run('B3LYP')
    _run('HF')
    _run('BLYP')
    again = _run('B3LYP')
    assert fresh.e_xc < -1.0
    assert again.e_xc == pytest.approx(fresh.e_xc, abs=1e-10)
    assert again.e_two_electron == pytest.approx(fresh.e_two_electron, abs=1e-10)


def test_components_sum_to_total():
    """total = e_electronic + nuclear_repulsion (+ e_dispersion when set);
    e_electronic = e_one_electron + e_two_electron + e_xc."""
    for r in (_run('HF'), _run('B3LYP'), _run('B3LYP', keywords={**LOOSE, 'D3': None})):
        disp = r.e_dispersion if 'e_dispersion' in r._values else 0.0
        assert r.total_energy == pytest.approx(
            r.e_electronic + r.nuclear_repulsion + disp, abs=1e-10)
        assert r.e_electronic == pytest.approx(
            r.e_one_electron + r.e_two_electron + r.e_xc, abs=1e-10)


# -- dispersion (EDISP) ---------------------------------------------------------

@pytest.mark.parametrize("keyword, ref_disp, ref_total", [
    ('D3', REF_D3_DISP, REF_D3_TOTAL),
    ('D3BJ', REF_D3BJ_DISP, REF_D3BJ_TOTAL),
])
def test_dispersion_matches_binary(keyword, ref_disp, ref_total):
    r = _run('B3LYP', keywords={**LOOSE, keyword: None})
    assert r.e_dispersion == pytest.approx(ref_disp, abs=1e-9)
    assert r.total_energy == pytest.approx(ref_total, abs=TOL)


def test_dispersion_is_part_of_total_energy():
    """Unlike e_external_charge, the dispersion correction IS in total_energy."""
    plain = _run('B3LYP')
    d3 = _run('B3LYP', keywords={**LOOSE, 'D3': None})
    assert d3.total_energy - plain.total_energy == pytest.approx(d3.e_dispersion, abs=1e-9)


def test_dispersion_gradient_matches_binary():
    r = pyquick.PyQuick('B3LYP', 'STO-3G', keywords={**TIGHT, 'D3': None},
                        log=False).get_grad(WATER)
    for got, ref in zip(r.gradient, REF_D3_GRADIENT):
        assert list(got) == pytest.approx(ref, abs=1e-6)


def test_dispersion_not_carried_into_next_run():
    _run('B3LYP', keywords={**LOOSE, 'D3BJ': None})
    r = _run('B3LYP', keywords={**LOOSE, 'D3': None})
    assert r.e_dispersion == pytest.approx(REF_D3_DISP, abs=1e-9)
    with pytest.raises(AttributeError):
        _ = _run('B3LYP').e_dispersion


# -- open shell -------------------------------------------------------------------

def test_open_shell_uhf_energy_matches_binary():
    r = _run('HF', charge=1, mult=2)
    assert r.total_energy == pytest.approx(REF_UHF_CATION_TOTAL, abs=TOL)
    assert r.metadata['calc_keyword'] == 'UHF'


def test_open_shell_udft_energy_matches_binary():
    r = _run('B3LYP', charge=1, mult=2)
    assert r.total_energy == pytest.approx(REF_UB3LYP_CATION_TOTAL, abs=TOL)
    assert r.metadata['calc_keyword'] == 'UDFT'


def test_open_shell_gradient_matches_binary():
    r = pyquick.PyQuick('HF', 'STO-3G', charge=1, mult=2, keywords=TIGHT,
                        log=False).get_grad(WATER)
    for got, ref in zip(r.gradient, REF_UHF_CATION_GRADIENT):
        assert list(got) == pytest.approx(ref, abs=1e-6)


def test_open_shell_geo_opt_matches_binary():
    r = pyquick.PyQuick('HF', 'STO-3G', charge=1, mult=2, keywords=TIGHT,
                        log=False).geo_opt(WATER)
    assert r.converged
    assert r.total_energy == pytest.approx(REF_UHF_CATION_OPT_TOTAL, abs=TOL)


def test_open_shell_charges_and_dipole_match_binary():
    r = _run('HF', charge=1, mult=2,
             properties=['mulliken_charges', 'lowdin_charges', 'dipole'])
    assert list(r.mulliken) == pytest.approx(REF_UHF_CATION_MULLIKEN, abs=1e-4)
    assert list(r.lowdin) == pytest.approx(REF_UHF_CATION_LOWDIN, abs=1e-4)
    assert list(r.dipole) == pytest.approx(REF_UHF_CATION_DIPOLE, abs=1e-4)
    assert sum(r.mulliken) == pytest.approx(1.0, abs=1e-6)


def test_uhf_closed_shell_equals_rhf():
    """UHF on a closed-shell molecule must reproduce RHF component by component."""
    rhf = _run('HF')
    uhf = _run('UHF')
    for attr in ('total_energy', 'e_electronic', 'nuclear_repulsion',
                 'e_one_electron', 'e_two_electron', 'e_xc'):
        assert getattr(uhf, attr) == pytest.approx(getattr(rhf, attr), abs=1e-7), attr


def test_impossible_multiplicity_raises_not_exits():
    """10 electrons cannot form a doublet.  QUICK reports this by calling
    exit(), which would kill the Python process (or Jupyter kernel); pyquick
    must catch it first and raise ValueError.  Run in a subprocess so a
    failure here cannot take down the test session."""
    import subprocess, sys, textwrap
    code = textwrap.dedent(f'''
        import pyquick
        try:
            pyquick.PyQuick('HF', 'STO-3G', mult=2, log=False).get_energy({WATER!r})
        except ValueError as e:
            print('VALUEERROR', e)
    ''')
    # Under mpiexec (MPI build), the child must not inherit this rank's MPI
    # identity: it would join the running job as a duplicate rank and corrupt
    # its shared-memory transport. Without the launcher's variables the child
    # starts as an independent single-rank run.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OMPI_', 'PMIX_', 'PRTE_', 'PMI_', 'HYDRA_'))}
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       timeout=120, env=env)
    assert p.returncode == 0, f"process died (exit {p.returncode}): {p.stdout[-300:]}"
    assert 'VALUEERROR' in p.stdout
    assert 'mult' in p.stdout.lower()


# -- ESP --------------------------------------------------------------------------

REF_ESP_CHARGES = [-0.522719, 0.261363, 0.261356]           # HF water, printed to 6 dp
REF_ESP_CHARGES_CATION = [-0.029993, 0.515017, 0.514976]    # UHF H2O+


def test_esp_charges_match_binary():
    r = _run('HF', properties=['esp_charges'])
    assert list(r.esp_charges) == pytest.approx(REF_ESP_CHARGES, abs=1e-5)
    assert sum(r.esp_charges) == pytest.approx(0.0, abs=1e-6)


def test_esp_charges_open_shell_match_binary():
    r = _run('HF', charge=1, mult=2, properties=['esp_charges'])
    assert list(r.esp_charges) == pytest.approx(REF_ESP_CHARGES_CATION, abs=1e-5)
    assert sum(r.esp_charges) == pytest.approx(1.0, abs=1e-6)


def test_esp_charges_from_get_grad():
    """The binary also fits ESP charges after a gradient job."""
    r = pyquick.PyQuick('HF', 'STO-3G', properties=['esp_charges'], keywords=LOOSE,
                        log=False).get_grad(WATER)
    assert list(r.esp_charges) == pytest.approx(REF_ESP_CHARGES, abs=1e-5)


def test_esp_charges_not_requested_raises():
    _run('HF', properties=['esp_charges'])           # earlier ESP run in this process
    r = _run('HF')
    with pytest.raises(AttributeError, match="esp_charges"):
        _ = r.esp_charges


def test_esp_with_log_false_leaves_no_files(tmp_path, monkeypatch):
    """log=False must leave nothing on disk, including ESP's surface file."""
    monkeypatch.chdir(tmp_path)
    _run('HF', properties=['esp_charges'])
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("keyword", ['ESP_CHARGE', 'esp_charge'])
def test_esp_charge_keyword_points_to_property(keyword):
    """ESP charges are requested with properties=['esp_charges'], not the raw
    keyword (which used to run silently and return nothing)."""
    with pytest.raises(ValueError, match="esp_charges"):
        pyquick.PyQuick('HF', 'STO-3G', keywords={keyword: None})


def test_esp_grid_keyword_rejected():
    """ESP on a user grid needs an external point file; not supported."""
    with pytest.raises(ValueError, match="(?i)esp_grid"):
        pyquick.PyQuick('HF', 'STO-3G', keywords={'ESP_GRID': None})


# ---------------------------------------------------------------------------
# Beta-spin results (unrestricted runs only)
#
# References: the quick binary's Molden export (EXPORT=MOLDEN) for UHF H2O+.
# Beta energies are its "Spin= Beta" Ene= values; the beta density is
# C_occ C_occ^T from its occupied beta orbitals.  The same construction from
# the alpha orbitals reproduces pyquick's alpha density_matrix to 1e-10.
# ---------------------------------------------------------------------------

REF_BETA_MO = [-20.9552272, -1.668309882, -1.076446361, -0.9529687494,
               -0.1830366227, 0.0233776459, 0.1322832866]
REF_BETA_DENSITY = [
    [1.05337889, -0.223302349, 0.03605546, -0.000404959, -0.025497093, -0.017306002, -0.017305041],
    [-0.223302349, 0.958009545, -0.19500748, 0.002190261, 0.137910639, 0.025217682, 0.025233392],
    [0.03605546, -0.19500748, 0.64242467, -0.005434358, -0.11795243, 0.020053875, 0.317555271],
    [-0.000404959, 0.002190261, -0.005434358, 6.1032e-05, 0.00384269, -0.001895839, -0.001895679],
    [-0.025497093, 0.137910639, -0.11795243, 0.00384269, 0.558997416, -0.329724142, 0.09101373],
    [-0.017306002, 0.025217682, 0.020053875, -0.001895839, -0.329724142, 0.20840851, -0.070739926],
    [-0.017305041, 0.025233392, 0.317555271, -0.001895679, 0.09101373, -0.070739926, 0.208411687],
]


def test_beta_mo_energies_open_shell_match_binary():
    r = _run('HF', charge=1, mult=2)
    assert r.beta_mo_energies.shape == (7,)
    assert list(r.beta_mo_energies) == pytest.approx(REF_BETA_MO, abs=1e-6)


def test_beta_density_open_shell_match_binary():
    r = _run('HF', charge=1, mult=2)
    assert r.beta_density_matrix.shape == (7, 7)
    for got, ref in zip(r.beta_density_matrix, REF_BETA_DENSITY):
        assert list(got) == pytest.approx(ref, abs=1e-6)


def test_beta_differs_from_alpha_open_shell():
    """A doublet has one more alpha than beta electron, so the spins differ."""
    for method in ('HF', 'B3LYP'):
        r = _run(method, charge=1, mult=2)
        assert r.beta_mo_energies.shape == r.mo_energies.shape
        assert r.beta_density_matrix.shape == r.density_matrix.shape
        assert not np.allclose(r.beta_mo_energies, r.mo_energies, atol=1e-4)
        assert not np.allclose(r.beta_density_matrix, r.density_matrix, atol=1e-4)


def test_uhf_closed_shell_beta_equals_alpha():
    """UHF on closed-shell water: beta must equal alpha (density_matrix is the
    alpha density on unrestricted runs)."""
    r = _run('UHF')
    assert np.allclose(r.beta_mo_energies, r.mo_energies, atol=1e-7)
    assert np.allclose(r.beta_density_matrix, r.density_matrix, atol=1e-7)


@pytest.mark.parametrize("method", ['HF', 'B3LYP'])
def test_beta_unavailable_for_restricted(method):
    """Restricted runs have no separate beta set (QUICK allocates a beta density
    for closed-shell DFT too, so this must not depend on allocation)."""
    r = _run(method)
    with pytest.raises(AttributeError, match="unrestricted"):
        _ = r.beta_mo_energies
    with pytest.raises(AttributeError, match="unrestricted"):
        _ = r.beta_density_matrix


def test_beta_not_carried_into_restricted_run():
    _run('HF', charge=1, mult=2)
    r = _run('HF')
    with pytest.raises(AttributeError, match="unrestricted"):
        _ = r.beta_density_matrix


def test_beta_in_persistent_job():
    calc = pyquick.PyQuick('HF', 'STO-3G', charge=1, mult=2, keywords=LOOSE, log=False)
    with calc.new_job(WATER) as job:
        r = job.run(WATER)
    assert list(r.beta_mo_energies) == pytest.approx(REF_BETA_MO, abs=1e-6)


def test_beta_properties_accepted_as_noop():
    """beta_* are always returned on unrestricted runs; naming them in
    properties is accepted (like mo_energies / density_matrix)."""
    calc = pyquick.PyQuick('HF', 'STO-3G', charge=1, mult=2,
                           properties=['beta_mo_energies', 'beta_density_matrix'])
    assert calc.properties == {'beta_mo_energies', 'beta_density_matrix'}


def test_density_matrix_meaning_documented():
    """density_matrix is the TOTAL density on restricted runs and the ALPHA
    density on unrestricted runs; the docstring must say so."""
    rhf = _run('HF')
    uhf = _run('UHF')
    assert np.allclose(rhf.density_matrix, 2 * uhf.density_matrix, atol=1e-7)
    doc = pyquick.Result.density_matrix.__doc__.lower()
    assert 'total' in doc and 'alpha' in doc and 'unrestricted' in doc


# ---------------------------------------------------------------------------
# Persistent DFT job with a moving geometry
#
# The DFT grid depends on the geometry.  job_step must free the previous
# step's grid before a new one is built (as QUICK's own API does between MD
# steps); without that the GPU build corrupted the heap once the grid size
# changed, crashing on job close.  The CPU build happened to survive, so on
# CPU this guards the energies; on a GPU build it also guards the crash.
# ---------------------------------------------------------------------------

WATER_STRETCHED = """
    O  -0.33840   0.00380   0.23923
    H  -0.33508  -0.00193  -0.83813
    H   0.67856  -0.00193   0.59530
"""


def test_persistent_dft_job_with_moving_geometry():
    calc = pyquick.PyQuick('B3LYP', 'STO-3G', keywords=LOOSE, log=False)
    geometries = [WATER, WATER_STRETCHED, WATER, WATER_STRETCHED]
    with calc.new_job(WATER) as job:
        steps = [job.run(g, gradient=(k == 3)) for k, g in enumerate(geometries)]
    # each step matches a one-shot run at the same geometry (also proves the
    # process survived closing the job)
    ref = {g: calc.get_energy(g).total_energy for g in (WATER, WATER_STRETCHED)}
    for g, r in zip(geometries, steps):
        assert r.total_energy == pytest.approx(ref[g], abs=TOL)
    assert steps[3].gradient.shape == (3, 3)


# ---------------------------------------------------------------------------
# Gradient on the external point charges (Result.point_charge_gradient)
#
# dE/dR of each charge, (ncharges, 3), Hartree/Bohr.  total_energy leaves out
# the charge-charge repulsion, so this gradient is the derivative of
# total_energy and does not contain that term either: finite differences of
# total_energy reproduce it, and the forces on atoms + charges sum to zero.
# ---------------------------------------------------------------------------

CLOSE_CHARGES = [(2.5, 0.5, 0.0, 0.6), (-1.0, 2.5, 0.5, -0.4)]   # near enough to polarize


def _pc_grad(method='HF', charges=CHARGES, **kw):
    return pyquick.PyQuick(method, 'STO-3G', keywords=TIGHT, log=False, **kw).get_grad(
        WATER, point_charges=charges)


def test_point_charge_gradient_shape():
    assert _pc_grad().point_charge_gradient.shape == (2, 3)
    assert _pc_grad(charges=[CHARGES[0]]).point_charge_gradient.shape == (1, 3)


@pytest.mark.parametrize("method, kw", [
    ('HF', {}),
    ('HF', {'charge': 1, 'mult': 2}),
    ('B3LYP', {}),
])
def test_point_charge_gradient_matches_finite_difference(method, kw):
    """Central finite differences of total_energy against moving each charge."""
    ANG_PER_BOHR = 0.52917721067
    h = 0.01                                # Angstrom
    calc = pyquick.PyQuick(method, 'STO-3G', keywords=TIGHT, log=False, **kw)
    analytic = calc.get_grad(WATER, point_charges=CLOSE_CHARGES).point_charge_gradient

    def energy(icharge, comp, delta):
        moved = [list(c) for c in CLOSE_CHARGES]
        moved[icharge][comp] += delta
        return calc.get_energy(WATER, point_charges=moved).total_energy

    for icharge in range(len(CLOSE_CHARGES)):
        for comp in range(3):
            fd = (energy(icharge, comp, h) - energy(icharge, comp, -h)) / (2 * h / ANG_PER_BOHR)
            assert analytic[icharge, comp] == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("method, abs_tol", [('HF', 1e-8), ('B3LYP', 1e-3)])
def test_forces_on_atoms_and_charges_sum_to_zero(method, abs_tol):
    """QM + charges is an isolated system (the DFT grid breaks this slightly)."""
    r = _pc_grad(method, charges=CLOSE_CHARGES)
    total = r.gradient.sum(axis=0) + r.point_charge_gradient.sum(axis=0)
    assert list(total) == pytest.approx([0.0, 0.0, 0.0], abs=abs_tol)


def test_point_charge_gradient_needs_get_grad():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    r = calc.get_energy(WATER, point_charges=CHARGES)
    with pytest.raises(AttributeError, match="get_grad"):
        _ = r.point_charge_gradient


def test_point_charge_gradient_needs_point_charges():
    r = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False).get_grad(WATER)
    with pytest.raises(AttributeError, match="point_charges"):
        _ = r.point_charge_gradient


def test_point_charge_gradient_not_carried_into_next_run():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    assert calc.get_grad(WATER, point_charges=CHARGES).point_charge_gradient.shape == (2, 3)
    assert calc.get_grad(WATER, point_charges=[CHARGES[0]]).point_charge_gradient.shape == (1, 3)
    with pytest.raises(AttributeError, match="point_charges"):
        _ = calc.get_grad(WATER).point_charge_gradient


def test_point_charge_gradient_in_persistent_job():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False)
    ref = calc.get_grad(WATER, point_charges=CHARGES).point_charge_gradient
    with calc.new_job(WATER, point_charges=CHARGES) as job:
        energy_step = job.run(WATER)
        grad_step = job.run(WATER, gradient=True)
    assert np.allclose(grad_step.point_charge_gradient, ref, atol=1e-6)
    with pytest.raises(AttributeError):
        _ = energy_step.point_charge_gradient


# ---------------------------------------------------------------------------
# Optimization trajectory (Result.trajectory_energies / trajectory_coordinates)
#
# References are the [GEOCONV] energies of the quick binary's Molden export on
# the same input.  Entry 0 is the input geometry, the last is the final one.
# QUICK stores Eel + Ecore per step, so these energies leave out any dispersion
# correction.
# ---------------------------------------------------------------------------

REF_TRAJ_DLFIND = [-74.94786381, -74.95030381, -74.96091494, -74.96560980,
                   -74.96588737, -74.96589936, -74.96590121, -74.96590121]
REF_TRAJ_LOPT = [-74.94786211, -74.92888436, -74.91224831, -74.96105792, -74.96414305,
                 -74.96562617, -74.96560561, -74.96585794, -74.96590087, -74.96590110]
METHANE = ("C 0 0 0\nH 0.6291 0.6291 0.6291\nH -0.6291 -0.6291 0.6291\n"
           "H -0.6291 0.6291 -0.6291\nH 0.6291 -0.6291 -0.6291")


def _opt(method='HF', keywords=LOOSE, geometry=WATER, max_cycles=None, **kw):
    return pyquick.PyQuick(method, 'STO-3G', keywords=keywords, log=False, **kw).geo_opt(
        geometry, max_cycles=max_cycles)


def test_trajectory_dlfind_matches_binary():
    r = _opt()
    assert list(r.trajectory_energies) == pytest.approx(REF_TRAJ_DLFIND, abs=1e-6)


def test_trajectory_lopt_matches_binary():
    r = _opt(keywords={**LOOSE, 'LOPT': None})
    assert list(r.trajectory_energies) == pytest.approx(REF_TRAJ_LOPT, abs=1e-6)


@pytest.mark.parametrize("keywords", [LOOSE, {**LOOSE, 'LOPT': None}], ids=['dlfind', 'lopt'])
def test_trajectory_shape_and_endpoints(keywords):
    r = _opt(keywords=keywords)
    n = len(r.trajectory_energies)
    assert r.trajectory_coordinates.shape == (n, 3, 3)
    assert np.allclose(r.trajectory_coordinates[0], r.coordinates, atol=1e-5)     # input
    assert np.allclose(r.trajectory_coordinates[-1], r.optimized_coordinates, atol=1e-5)
    assert r.trajectory_energies[-1] == pytest.approx(r.total_energy, abs=TOL)


def test_trajectory_open_shell():
    r = _opt(charge=1, mult=2)
    assert len(r.trajectory_energies) == 7                       # binary: 7 steps
    assert r.trajectory_energies[-1] == pytest.approx(REF_UHF_CATION_OPT_TOTAL, abs=TOL)


def test_trajectory_dft():
    r = _opt('B3LYP')
    assert r.trajectory_energies[-1] == pytest.approx(r.total_energy, abs=TOL)
    assert r.trajectory_energies[0] == pytest.approx(REF_B3LYP_TOTAL, abs=TOL)


def test_trajectory_energies_leave_out_dispersion():
    r = _opt('B3LYP', keywords={**LOOSE, 'D3': None})
    assert r.trajectory_energies[-1] + r.e_dispersion == pytest.approx(r.total_energy, abs=TOL)


def test_trajectory_respects_max_cycles():
    r = _opt(max_cycles=2)
    assert not r.converged
    assert len(r.trajectory_energies) <= 3


@pytest.mark.parametrize("run", ['get_energy', 'get_grad'])
def test_trajectory_only_from_geo_opt(run):
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    r = getattr(calc, run)(WATER)
    with pytest.raises(AttributeError, match="geo_opt"):
        _ = r.trajectory_energies
    with pytest.raises(AttributeError, match="geo_opt"):
        _ = r.trajectory_coordinates


def test_lopt_choice_not_carried_into_next_optimization():
    """QUICK does not reset its optimizer choice between runs in one process."""
    assert len(_opt().trajectory_energies) == 8                       # DL-Find
    assert len(_opt(keywords={**LOOSE, 'LOPT': None}).trajectory_energies) == 10
    assert len(_opt().trajectory_energies) == 8                       # DL-Find again


def test_trajectory_not_carried_between_runs():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    assert len(calc.geo_opt(WATER).trajectory_energies) == 8
    with pytest.raises(AttributeError):
        _ = calc.get_energy(WATER).trajectory_energies


def test_trajectory_across_molecule_sizes():
    """Sizes change between optimizations, in both directions and on the UHF
    path (which also writes MOs through the Molden code)."""
    kw = {**LOOSE, 'LOPT': None}
    assert _opt(keywords=kw, charge=1, mult=2).trajectory_coordinates.shape[1:] == (3, 3)
    r = _opt(keywords=kw, geometry=METHANE, charge=1, mult=2, max_cycles=3)
    assert r.trajectory_coordinates.shape[1:] == (5, 3)
    # Known QUICK bug, not pyquick's: the plain quick.cuda.MPI binary also crashes on
    # a gradient of a molecule with s functions only (cudaMemcpy illegal access).
    if not (pyquick.mpi_enabled() and pyquick.gpu_enabled()):
        r = _opt(keywords=kw, geometry=H2)
        assert r.trajectory_coordinates.shape[1:] == (2, 3)


@pytest.mark.parametrize("log", [False, True])
def test_geo_opt_leaves_no_molden_file(tmp_path, monkeypatch, log):
    """The trajectory borrows QUICK's Molden storage, never its file output."""
    monkeypatch.chdir(tmp_path)
    for kw in ({}, {'charge': 1, 'mult': 2}):
        pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=log, **kw).geo_opt(WATER)
    left = sorted(os.listdir(tmp_path))
    assert not [f for f in left if 'molden' in f.lower() or f.startswith('fort.')], left
    if not log:
        assert left == []


def test_energy_run_after_geo_opt_still_works():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    calc.geo_opt(WATER)
    assert calc.get_energy(WATER).total_energy == pytest.approx(REF_ENERGY, abs=TOL)


# ---------------------------------------------------------------------------
# MO coefficients, occupations, overlap matrix, AO labels
#
# Orbital signs are arbitrary (and differ on CUDA), so everything here is
# sign-independent: densities rebuilt from occupied orbitals, orthonormality,
# Mulliken charges rebuilt through ao_labels, and |C| against the binary's
# Molden export (UHF H2O+, first alpha orbital).
# ---------------------------------------------------------------------------

REF_ALPHA_MO1_ABS = [0.9942744458, 0.0249409997, 0.0025914814, 0.0000291075,
                     0.0018332622, 0.0049040562, 0.0049027141]
Z = {'H': 1.0, 'C': 6.0, 'O': 8.0}


def _occupied(c, occupations):
    return c[:, occupations > 0.5]


def test_mo_coefficients_shape():
    r = _run('HF')
    assert r.mo_coefficients.shape == (7, 7)              # (nbasis, nmo): columns are MOs
    assert r.mo_occupations.shape == (7,)


def test_mo_occupations_restricted():
    assert list(_run('HF').mo_occupations) == [2.0] * 5 + [0.0] * 2


def test_mo_occupations_unrestricted():
    r = _run('HF', charge=1, mult=2)
    assert list(r.mo_occupations) == [1.0] * 5 + [0.0] * 2
    assert list(r.beta_mo_occupations) == [1.0] * 4 + [0.0] * 3


@pytest.mark.parametrize("method", ['HF', 'B3LYP'])
def test_density_from_occupied_orbitals_restricted(method):
    r = _run(method)
    c = _occupied(r.mo_coefficients, r.mo_occupations)
    assert np.allclose(2.0 * c @ c.T, r.density_matrix, atol=1e-7)


@pytest.mark.parametrize("method", ['HF', 'B3LYP'])
def test_density_from_occupied_orbitals_unrestricted(method):
    r = _run(method, charge=1, mult=2)
    a = _occupied(r.mo_coefficients, r.mo_occupations)
    b = _occupied(r.beta_mo_coefficients, r.beta_mo_occupations)
    assert np.allclose(a @ a.T, r.density_matrix, atol=1e-7)
    assert np.allclose(b @ b.T, r.beta_density_matrix, atol=1e-7)


def test_mo_coefficients_match_binary_up_to_sign():
    r = _run('HF', charge=1, mult=2)
    assert list(np.abs(r.mo_coefficients[:, 0])) == pytest.approx(REF_ALPHA_MO1_ABS, abs=1e-6)


def test_mo_coefficients_are_orthonormal():
    r = _run('HF')
    c, s = r.mo_coefficients, r.overlap_matrix
    assert np.allclose(c.T @ s @ c, np.eye(7), atol=1e-8)


def test_overlap_matrix_properties():
    s = _run('HF').overlap_matrix
    assert s.shape == (7, 7)
    assert np.allclose(s, s.T, atol=1e-12)
    assert list(np.diag(s)) == pytest.approx([1.0] * 7, abs=1e-8)


def test_electron_count_from_density_and_overlap():
    r = _run('HF')
    assert np.trace(r.density_matrix @ r.overlap_matrix) == pytest.approx(10.0, abs=1e-6)
    r = _run('HF', charge=1, mult=2)
    total = r.density_matrix + r.beta_density_matrix
    assert np.trace(total @ r.overlap_matrix) == pytest.approx(9.0, abs=1e-6)


def test_ao_labels_water_sto3g():
    assert _run('HF').ao_labels == [
        (0, 'O', 's'), (0, 'O', 's'), (0, 'O', 'px'), (0, 'O', 'py'), (0, 'O', 'pz'),
        (1, 'H', 's'), (2, 'H', 's')]


def test_ao_labels_cartesian_d_functions():
    r = pyquick.PyQuick('HF', '6-31G*', keywords=LOOSE, log=False).get_energy(WATER)
    assert len(r.ao_labels) == r.mo_coefficients.shape[0] == 19
    d = [t for atom, sym, t in r.ao_labels if t[0] == 'd']
    assert sorted(d) == sorted(['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'])


def test_mulliken_rebuilt_through_ao_labels():
    """Population per atom = sum of diag(PS) over that atom's basis functions."""
    r = _run('HF', properties=['mulliken_charges'])
    pop = np.diag(r.density_matrix @ r.overlap_matrix)
    natoms = len(r.atomic_numbers)
    charges = list(r.atomic_numbers.astype(float))
    for (atom, _sym, _kind), p in zip(r.ao_labels, pop):
        charges[atom] -= p
    assert charges == pytest.approx(list(r.mulliken), abs=1e-5)
    assert len(charges) == natoms


def test_beta_mo_coefficients_unavailable_for_restricted():
    r = _run('HF')
    with pytest.raises(AttributeError, match="unrestricted"):
        _ = r.beta_mo_coefficients
    with pytest.raises(AttributeError, match="unrestricted"):
        _ = r.beta_mo_occupations


def test_mo_coefficients_after_geo_opt_match_final_density():
    r = _opt()
    c = _occupied(r.mo_coefficients, r.mo_occupations)
    assert np.allclose(2.0 * c @ c.T, r.density_matrix, atol=1e-6)


def test_mo_coefficients_in_persistent_job():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False)
    with calc.new_job(WATER) as job:
        job.run(WATER)
        r = job.run(WATER_STRETCHED)
    c = _occupied(r.mo_coefficients, r.mo_occupations)
    assert np.allclose(2.0 * c @ c.T, r.density_matrix, atol=1e-7)


def test_mo_arrays_follow_the_molecule_size():
    assert _run('HF', geometry=H2).mo_coefficients.shape == (2, 2)
    r = _run('HF')
    assert r.mo_coefficients.shape == (7, 7)
    assert r.overlap_matrix.shape == (7, 7)
    assert len(r.ao_labels) == 7


def test_larger_basis_after_smaller_one():
    """QUICK sizes some tables once per process (under MPI, its load-balancing
    tables); a later, larger run must not overrun them."""
    def energy(basis, geometry=WATER):
        return pyquick.PyQuick('HF', basis, keywords=LOOSE, log=False).get_energy(
            geometry).total_energy
    small = energy('STO-3G', H2)
    large = energy('6-31G*')
    assert energy('6-31G*') == pytest.approx(large, abs=1e-8)
    assert energy('STO-3G', H2) == pytest.approx(small, abs=1e-8)
    assert energy('6-31G*') == pytest.approx(large, abs=1e-8)


def test_new_always_on_names_accepted_in_properties():
    names = ['mo_coefficients', 'mo_occupations', 'beta_mo_coefficients',
             'beta_mo_occupations', 'overlap_matrix', 'ao_labels']
    calc = pyquick.PyQuick('HF', 'STO-3G', properties=names)
    assert calc.properties == set(names)


# ---------------------------------------------------------------------------
# Kinetic vs nuclear-attraction split of e_one_electron
#
# QUICK does not print these, so references come from PySCF on the same inputs
# (its total energies agree with QUICK's to ~2e-8; d functions are Cartesian).
# 6-31G* needs QUICK's own basis file there: PySCF's copy differs slightly, which
# barely moves the energy but shifts T and V by ~1e-5.
# T and V are first-order in the density, unlike the total energy, so these
# comparisons need a tightly converged SCF (TIGHT).
# The split is exact: e_kinetic + e_nuclear_attraction = e_one_electron.
# ---------------------------------------------------------------------------

REF_T_WATER, REF_V_WATER = 74.379235552, -194.987647603
REF_T_CATION, REF_V_CATION = 72.931488501, -187.971278133
REF_T_631GS, REF_V_631GS = 75.437464045, -196.701521176    # with QUICK's basis file
REF_T_H2P, REF_V_H2P = 0.600643589, -1.853953375
REF_T_PC, REF_V_PC = 74.342682280, -194.856139946    # with CHARGES, nuclei only
REF_V_PC_CHARGES = -1.852708119                    # electrons <-> CHARGES


@pytest.mark.parametrize("kw, ref_t, ref_v", [
    ({}, REF_T_WATER, REF_V_WATER),
    ({'charge': 1, 'mult': 2}, REF_T_CATION, REF_V_CATION),
    ({'charge': 1, 'mult': 2, 'geometry': H2}, REF_T_H2P, REF_V_H2P),
], ids=['rhf', 'uhf', 'one-electron'])
def test_kinetic_and_attraction_match_pyscf(kw, ref_t, ref_v):
    r = _run('HF', keywords=TIGHT, **kw)
    assert r.e_kinetic == pytest.approx(ref_t, abs=1e-5)
    assert r.e_nuclear_attraction == pytest.approx(ref_v, abs=1e-5)


def test_kinetic_cartesian_d_functions_match_pyscf():
    r = pyquick.PyQuick('HF', '6-31G*', keywords=TIGHT, log=False).get_energy(WATER)
    assert r.e_kinetic == pytest.approx(REF_T_631GS, abs=1e-5)
    assert r.e_nuclear_attraction == pytest.approx(REF_V_631GS, abs=1e-5)


def test_one_electron_energy_is_whole_under_mpi():
    """Each MPI rank holds part of E1e; the Result must carry the sum."""
    assert _run('HF', keywords=TIGHT).e_one_electron == pytest.approx(
        REF_T_WATER + REF_V_WATER, abs=1e-5)


@pytest.mark.parametrize("method, kw", [('HF', {}), ('B3LYP', {}), ('B3LYP', {'charge': 1, 'mult': 2})])
def test_split_adds_up_to_one_electron_energy(method, kw):
    r = _run(method, **kw)
    assert r.e_kinetic + r.e_nuclear_attraction == pytest.approx(r.e_one_electron, abs=1e-10)
    assert r.e_kinetic > 0 and r.e_nuclear_attraction < 0


def test_attraction_includes_point_charges():
    """e_nuclear_attraction also holds the electrons' attraction to the charges."""
    r = pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False).get_energy(
        WATER, point_charges=CHARGES)
    assert r.e_kinetic == pytest.approx(REF_T_PC, abs=1e-5)
    assert r.e_nuclear_attraction == pytest.approx(REF_V_PC + REF_V_PC_CHARGES, abs=1e-5)


def test_split_from_get_grad_matches_get_energy():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False)
    assert calc.get_grad(WATER).e_kinetic == pytest.approx(REF_T_WATER, abs=1e-5)


def test_split_after_geo_opt_is_for_the_final_geometry():
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False)
    r = calc.geo_opt(WATER)
    final = "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in
                      zip(['O', 'H', 'H'], r.optimized_coordinates))
    assert r.e_kinetic == pytest.approx(calc.get_energy(final).e_kinetic, abs=1e-4)


def test_split_in_persistent_job_leaves_later_steps_intact():
    """Rebuilding the kinetic matrix borrows QUICK's Fock array; steps must not notice."""
    calc = pyquick.PyQuick('HF', 'STO-3G', keywords=TIGHT, log=False)
    with calc.new_job(WATER) as job:
        job.run(WATER_STRETCHED)
        r = job.run(WATER, gradient=True)
    assert r.total_energy == pytest.approx(REF_ENERGY, abs=TOL)
    assert r.e_kinetic == pytest.approx(REF_T_WATER, abs=1e-5)


def test_split_does_not_disturb_properties():
    r = _run('HF', properties=['mulliken_charges', 'dipole', 'esp_charges'])
    assert sum(r.mulliken) == pytest.approx(0.0, abs=1e-6)
    assert list(r.esp_charges) == pytest.approx(REF_ESP_CHARGES, abs=1e-5)
