"""
pytest unit + integration tests for the pyquick PyQuick/Result API.

Pure-Python tests (validation, method resolution, input assembly) need only that
the compiled extension imports.  The integration tests additionally run a small
HF/STO-3G water job.

Job type is chosen by which method you call:
    calc.get_energy(geometry)  -> single-point energy
    calc.get_grad(geometry)    -> energy + forces        (Phase 2, not yet implemented)
    calc.geo_opt(geometry)     -> optimized geometry      (Phase 2, not yet implemented)
`properties` selects only optional extras (charges, MO energies, density matrix).
"""

import math

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


def test_deferred_property_reports_phase():
    with pytest.raises(ValueError) as excinfo:
        pyquick.PyQuick(method='HF', basis='STO-3G', properties=['esp_charges'])
    assert 'Phase 4' in str(excinfo.value)


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
    """dispersion only present if requested; external charge only if EXTCHARGES set.

    The AttributeError must explain *why* the value is unavailable and how to enable it.
    """
    r = _energy()   # no D* keyword, no EXTCHARGES
    with pytest.raises(AttributeError, match="(?i)dispersion"):
        _ = r.e_dispersion
    with pytest.raises(AttributeError, match="EXTCHARGES"):
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
