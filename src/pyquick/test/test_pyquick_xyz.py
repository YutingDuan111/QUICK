"""
Reading geometries from .xyz files.  Run from this directory:

    python -m pytest test_pyquick_xyz.py -q

An .xyz file is:

    <number of atoms>
    <comment line, may be empty>
    SYMBOL  X  Y  Z      (Angstrom; columns after Z are ignored)
    ...

Wherever a geometry is accepted, a path works too: a pathlib.Path, or a string
with no line break that ends in .xyz.  pyquick.read_xyz(path) returns the
geometry as a 'SYMBOL X Y Z' string.
"""

import pathlib

import pytest

pyquick = pytest.importorskip("pyquick")

WATER = """
    O  -0.33840   0.00380   0.23923
    H  -0.33510  -0.00190  -0.83277
    H   0.67350  -0.00190   0.59353
"""
WATER_XYZ = """3
water, HF/STO-3G reference geometry
O  -0.33840   0.00380   0.23923
H  -0.33510  -0.00190  -0.83277
H   0.67350  -0.00190   0.59353
"""
LOOSE = {'cutoff': '1.0e-9', 'denserms': '1.0e-6'}
REF_HF_ENERGY = -74.947863811        # same reference as test_pyquick_api.py
TOL = 4.0e-5


def _calc(**kw):
    return pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False, **kw)


@pytest.fixture
def mol_xyz(tmp_path, monkeypatch):
    """mol.xyz in the working directory; the fixture value is its relative name."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mol.xyz").write_text(WATER_XYZ)
    return "mol.xyz"


# ---------------------------------------------------------------------------
# read_xyz
# ---------------------------------------------------------------------------

def _atoms(geometry):
    rows = [line.split() for line in geometry.splitlines() if line.strip()]
    return [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in rows]


def test_read_xyz_returns_atom_lines(mol_xyz):
    assert _atoms(pyquick.read_xyz(mol_xyz)) == _atoms(WATER)


def test_read_xyz_accepts_path_object(mol_xyz):
    assert _atoms(pyquick.read_xyz(pathlib.Path(mol_xyz))) == _atoms(WATER)


def test_read_xyz_empty_comment_line(tmp_path):
    path = tmp_path / "blank.xyz"
    path.write_text("1\n\nH 0.0 0.0 0.0\n")
    assert _atoms(pyquick.read_xyz(path)) == [('H', 0.0, 0.0, 0.0)]


def test_read_xyz_comment_that_looks_like_an_atom(tmp_path):
    """The comment line is skipped by position, whatever it contains."""
    path = tmp_path / "tricky.xyz"
    path.write_text("1\nO 1.0 2.0 3.0\nH 0.0 0.0 0.0\n")
    assert _atoms(pyquick.read_xyz(path)) == [('H', 0.0, 0.0, 0.0)]


def test_read_xyz_ignores_extra_columns_and_trailing_blank_lines(tmp_path):
    path = tmp_path / "extra.xyz"
    path.write_text("2\nforces\nH 0 0 0 0.1 0.2 0.3\nH 0 0 0.74 -0.1 -0.2 -0.3\n\n\n")
    assert _atoms(pyquick.read_xyz(path)) == [('H', 0.0, 0.0, 0.0), ('H', 0.0, 0.0, 0.74)]


@pytest.mark.parametrize("content,message", [
    ("",                                 "atom count"),   # empty file
    ("two\ncomment\nH 0 0 0\n",          "atom count"),   # first line not a number
    ("0\ncomment\n",                     "atom count"),   # no atoms
    ("2\ncomment\nH 0 0 0\n",            "2 atoms"),      # fewer atoms than declared
    ("1\ncomment\nH 0 0\n",              "SYMBOL X Y Z"), # missing coordinate
    ("1\ncomment\nH 0 0 zero\n",         "SYMBOL X Y Z"), # coordinate not a number
    ("1\nc\nH 0 0 0\n1\nc\nH 0 0 1\n",   "one geometry"), # several frames
])
def test_read_xyz_rejects_bad_files(tmp_path, content, message):
    path = tmp_path / "bad.xyz"
    path.write_text(content)
    with pytest.raises(ValueError, match=message) as err:
        pyquick.read_xyz(path)
    assert "bad.xyz" in str(err.value)        # the error names the file


def test_read_xyz_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        pyquick.read_xyz("nothing_here.xyz")


# ---------------------------------------------------------------------------
# every calculation takes a path
# ---------------------------------------------------------------------------

def test_get_energy_from_xyz_file(mol_xyz):
    assert _calc().get_energy(mol_xyz).total_energy == pytest.approx(REF_HF_ENERGY, abs=TOL)


def test_get_energy_from_path_object(mol_xyz):
    energy = _calc().get_energy(pathlib.Path(mol_xyz)).total_energy
    assert energy == pytest.approx(REF_HF_ENERGY, abs=TOL)


def test_uppercase_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "MOL.XYZ").write_text(WATER_XYZ)
    assert _calc().get_energy("MOL.XYZ").total_energy == pytest.approx(REF_HF_ENERGY, abs=TOL)


def test_file_and_string_give_the_same_result(mol_xyz):
    from_file = _calc().get_grad(mol_xyz)
    from_string = _calc().get_grad(WATER)
    assert from_file.total_energy == from_string.total_energy
    # GPU sums vary at 1e-15 between identical runs, so the gradient is not exact
    assert from_file.gradient == pytest.approx(from_string.gradient, abs=1e-10)
    assert (from_file.coordinates == from_string.coordinates).all()


def test_geo_opt_from_xyz_file(mol_xyz):
    from_file = _calc().geo_opt(mol_xyz)
    from_string = _calc().geo_opt(WATER)
    assert from_file.total_energy == pytest.approx(from_string.total_energy, abs=1e-8)


def test_charge_and_mult_are_checked_for_a_file(mol_xyz):
    """The spin check must see the atoms in the file, not the file's name."""
    with pytest.raises(ValueError, match="impossible"):
        _calc(mult=2).get_energy(mol_xyz)


def test_geo_opt_atom_guard_sees_the_file(tmp_path, monkeypatch):
    """DL-Find cannot do 2 atoms; the guard counts the atoms in the file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "h2.xyz").write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    with pytest.raises(ValueError, match="DL-Find"):
        _calc().geo_opt("h2.xyz")


def test_persistent_job_from_xyz_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.xyz").write_text(WATER_XYZ)
    (tmp_path / "b.xyz").write_text(WATER_XYZ)
    with _calc().new_job("a.xyz") as job:
        energy = job.run("b.xyz").total_energy
    assert energy == pytest.approx(REF_HF_ENERGY, abs=TOL)


# ---------------------------------------------------------------------------
# strings that are not paths keep working as before
# ---------------------------------------------------------------------------

def test_geometry_string_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _calc().get_energy(WATER).total_energy == pytest.approx(REF_HF_ENERGY, abs=TOL)


def test_one_atom_string_is_not_a_path():
    """A one-line geometry has no line break but does not end in .xyz."""
    energy = pyquick.PyQuick('HF', 'STO-3G', keywords=LOOSE, log=False,
                             mult=2).get_energy("H 0.0 0.0 0.0").total_energy
    assert energy < 0.0


def test_missing_xyz_file_in_a_calculation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _calc().get_energy("nothing_here.xyz")
