"""
How a script picks a pyquick mode.  One install holds up to four engines and
the import chooses one:

    import pyquick              serial; the engine loads on first use
    import pyquick.serial       serial, also under mpiexec (task farming)
    import pyquick.mpi          MPI        (cmake -DMPI=TRUE)
    import pyquick.gpu          GPU        (-DCUDA=TRUE or -DHIP=TRUE)
    import pyquick.gpu_mpi      GPU + MPI  (both)

Run with PLAIN pytest; every case starts its own Python process, because a
process can use only one engine:

    python -m pytest test_pyquick_modes.py -q

Optional environment variables:
  PYQUICK_MPIEXEC   launcher command, default "mpiexec" ("mpiexec --oversubscribe"
                    on a machine with fewer cores than ranks)
  PYQUICK_NGPUS     GPUs the launched jobs may use; default: the entries of
                    CUDA_VISIBLE_DEVICES, else the GPUs `nvidia-smi -L` lists

Which modes the install has is read from its layout (pyquick/_engines/<mode>),
not by importing them, so a missing engine is checked, not assumed.
"""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest

MODES = ('serial', 'mpi', 'gpu', 'gpu_mpi')
MODULES = {'serial': 'pyquick', 'mpi': 'pyquick.mpi',
           'gpu': 'pyquick.gpu', 'gpu_mpi': 'pyquick.gpu_mpi'}
FLAGS = {'serial': (False, False), 'mpi': (True, False),
         'gpu': (False, True), 'gpu_mpi': (True, True)}     # (mpi_enabled, gpu_enabled)
BUILT_WITH = {'mpi': '-DMPI=TRUE', 'gpu': '-DCUDA=TRUE',
              'gpu_mpi': '-DCUDA=TRUE -DMPI=TRUE'}

MPIEXEC = shlex.split(os.environ.get("PYQUICK_MPIEXEC", "mpiexec"))
TIMEOUT = 300
_LAUNCHER_PREFIXES = ('OMPI_', 'PMIX_', 'PRTE_', 'PMI_', 'HYDRA_')

WATER = "O -0.33840 0.00380 0.23923\nH -0.33510 -0.00190 -0.83277\nH 0.67350 -0.00190 0.59353"
REF_HF_ENERGY = -74.947863811        # HF/STO-3G water, same reference as test_pyquick_api.py
TOL = 4.0e-5
TOL_MODES = 1.0e-6                   # between modes: GPU sums use atomics

if any(k in os.environ for k in ('OMPI_COMM_WORLD_SIZE', 'PMIX_RANK', 'PMI_RANK', 'PMI_SIZE')):
    pytest.skip("this file starts its own processes: run it with plain pytest, "
                "not under mpiexec", allow_module_level=True)

_SPEC = importlib.util.find_spec("pyquick")
if _SPEC is None:
    pytest.skip("pyquick is not installed (is quick.rc sourced?)", allow_module_level=True)
PACKAGE_DIR = list(_SPEC.submodule_search_locations)[0]


def _built(mode):
    """True if the install has this mode's engine (pyquick/_engines/<mode>/_pyquick*)."""
    folder = os.path.join(PACKAGE_DIR, '_engines', mode)
    return os.path.isdir(folder) and any(f.startswith('_pyquick') for f in os.listdir(folder))


def _visible_gpus():
    if os.environ.get("PYQUICK_NGPUS"):
        return int(os.environ["PYQUICK_NGPUS"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip()])
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        return sum(1 for line in p.stdout.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.TimeoutExpired):
        return 0


BUILT = [m for m in MODES if _built(m)]
NGPU = _visible_gpus() if any(m in BUILT for m in ('gpu', 'gpu_mpi')) else 0
HAVE_MPIEXEC = shutil.which(MPIEXEC[0]) is not None


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
    env.update(extra)
    return env


def _run(tmp_path, code, nranks=None, mpi4py=True):
    """Run *code* in a fresh interpreter, or under mpiexec -n nranks."""
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(code))
    if nranks is None:
        cmd = [sys.executable, str(script)]
    elif mpi4py:
        cmd = MPIEXEC + ["-n", str(nranks), sys.executable, "-m", "mpi4py", str(script)]
    else:
        cmd = MPIEXEC + ["-n", str(nranks), sys.executable, str(script)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path,
                          env=_clean_env(), timeout=TIMEOUT)


def _ok(proc):
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    return proc.stdout


def _need(mode, run=False):
    """Skip unless *mode* is built (and, with run=True, can run here)."""
    if mode not in BUILT:
        pytest.skip(f"this install has no {mode} engine")
    if run and mode in ('gpu', 'gpu_mpi') and NGPU == 0:
        pytest.skip("no visible GPU")


ENERGY = """
    import json
    import {module} as pyquick
    r = pyquick.PyQuick('HF', 'STO-3G', log=False,
                        keywords={{'cutoff': '1.0e-9', 'denserms': '1.0e-6'}}).get_energy({water!r})
    if pyquick.mpi_rank() == 0:
        print('RESULT ' + json.dumps({{'energy': r.total_energy, 'size': pyquick.mpi_size()}}))
"""


def _energy(tmp_path, module, nranks=None):
    """Water HF energy with *module*; mpi4py starts only for the MPI engines."""
    mpi4py = module in ('pyquick.mpi', 'pyquick.gpu_mpi')
    out = _ok(_run(tmp_path, ENERGY.format(module=module, water=WATER), nranks=nranks,
                   mpi4py=mpi4py))
    lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
    assert len(lines) == 1, f"expected one RESULT line (rank 0 only), got {lines}"
    return json.loads(lines[0][len("RESULT "):])


# ---------------------------------------------------------------------------
# which engines exist, and what each reports
# ---------------------------------------------------------------------------

def test_serial_engine_is_always_built():
    assert 'serial' in BUILT, f"no serial engine under {PACKAGE_DIR}/_engines"


@pytest.mark.parametrize("mode", MODES)
def test_mode_reports_itself(tmp_path, mode):
    _need(mode)
    out = _ok(_run(tmp_path, f"""
        import {MODULES[mode]} as pyquick
        print('FLAGS', pyquick.mpi_enabled(), pyquick.gpu_enabled(), pyquick.mpi_size())
    """))
    mpi, gpu = FLAGS[mode]
    assert f"FLAGS {mpi} {gpu} 1" in out


@pytest.mark.parametrize("mode", ['mpi', 'gpu', 'gpu_mpi'])
def test_mode_not_built_raises(tmp_path, mode):
    if mode in BUILT:
        pytest.skip(f"this install has the {mode} engine")
    out = _ok(_run(tmp_path, f"""
        try:
            import {MODULES[mode]}
        except ImportError as e:
            print('IMPORTERROR', e)
    """))
    assert f"IMPORTERROR pyquick.{mode} is not available" in out
    assert "built without" in out


# ---------------------------------------------------------------------------
# engines load lazily, and only one per process
# ---------------------------------------------------------------------------

LOADED = "any(k.startswith('pyquick._engines') for k in sys.modules)"


def test_plain_import_loads_no_engine(tmp_path):
    (tmp_path / "mol.xyz").write_text("3\nwater\n" + WATER + "\n")
    out = _ok(_run(tmp_path, f"""
        import sys
        import pyquick
        print('AFTER IMPORT', {LOADED})
        print('ATOMS', len(pyquick.read_xyz('mol.xyz').splitlines()))
        print('AFTER READ_XYZ', {LOADED})
        pyquick.PyQuick
        print('AFTER FIRST USE', {LOADED})
    """))
    assert "AFTER IMPORT False" in out
    assert "ATOMS 3" in out
    assert "AFTER READ_XYZ False" in out
    assert "AFTER FIRST USE True" in out


@pytest.mark.parametrize("mode", ['mpi', 'gpu', 'gpu_mpi'])
def test_mode_import_does_not_load_serial(tmp_path, mode):
    """`import pyquick.mpi` runs pyquick/__init__.py first; that must not load
    the serial engine as well."""
    _need(mode)
    out = _ok(_run(tmp_path, f"""
        import sys
        import {MODULES[mode]}
        print('SERIAL LOADED', any(k.startswith('pyquick._engines.serial') for k in sys.modules))
    """))
    assert "SERIAL LOADED False" in out


@pytest.mark.parametrize("first,second", [(a, b) for a in MODES for b in MODES if a != b])
def test_second_mode_in_one_process_raises(tmp_path, first, second):
    _need(first)
    _need(second)
    out = _ok(_run(tmp_path, f"""
        import {MODULES[first]} as one
        one.PyQuick
        try:
            import {MODULES[second]} as two
            two.PyQuick
        except ImportError as e:
            print('IMPORTERROR', e)
    """))
    assert f"IMPORTERROR pyquick is already using the {first} engine" in out


def test_serial_spelled_two_ways_is_one_engine(tmp_path):
    out = _ok(_run(tmp_path, """
        import pyquick
        import pyquick.serial
        print('SAME', pyquick.PyQuick is pyquick.serial.PyQuick)
    """))
    assert "SAME True" in out


# ---------------------------------------------------------------------------
# plain `import pyquick` under mpiexec (the D5 guard)
# ---------------------------------------------------------------------------

needs_mpiexec = pytest.mark.skipif(not HAVE_MPIEXEC, reason=f"{MPIEXEC[0]} not on PATH")


@needs_mpiexec
def test_plain_import_under_mpiexec_raises(tmp_path):
    """An old MPI script run unchanged must stop, not run serial on every rank.
    Plain python (no -m mpi4py): each rank fails on its own."""
    proc = _run(tmp_path, """
        import pyquick
        pyquick.PyQuick('HF', 'STO-3G', log=False)
    """, nranks=2, mpi4py=False)
    assert proc.returncode != 0
    text = proc.stdout + proc.stderr
    assert "running under mpiexec with 2 ranks" in text
    assert "pyquick.mpi" in text and "pyquick.gpu_mpi" in text


@needs_mpiexec
def test_parent_import_under_mpiexec_is_not_guarded(tmp_path):
    """`import pyquick.mpi` imports the pyquick package first; that alone must
    not trip the guard."""
    _need('mpi')
    assert _energy(tmp_path, 'pyquick.mpi', nranks=2)['size'] == 2


@needs_mpiexec
def test_plain_import_with_one_rank_runs_serial(tmp_path):
    res = _energy(tmp_path, 'pyquick', nranks=1)
    assert res['energy'] == pytest.approx(REF_HF_ENERGY, abs=TOL)
    assert res['size'] == 1


@needs_mpiexec
def test_serial_module_under_mpiexec_runs_on_every_rank(tmp_path):
    """Task farming: `import pyquick.serial` asks for serial on purpose, so the
    guard lets every rank run its own calculation."""
    out = _ok(_run(tmp_path, f"""
        import pyquick.serial as pyquick
        r = pyquick.PyQuick('HF', 'STO-3G', log=False,
                            keywords={{'cutoff': '1.0e-9', 'denserms': '1.0e-6'}}).get_energy({WATER!r})
        print('ENERGY', r.total_energy, pyquick.mpi_size())
    """, nranks=2, mpi4py=False))
    lines = [l.split() for l in out.splitlines() if l.startswith("ENERGY ")]
    assert len(lines) == 2, f"expected one line per rank, got {lines}"
    for _, energy, size in lines:
        assert float(energy) == pytest.approx(REF_HF_ENERGY, abs=TOL)
        assert size == '1'


# ---------------------------------------------------------------------------
# the same script gives the same energy in every mode
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def serial_energy(tmp_path_factory):
    return _energy(tmp_path_factory.mktemp("serial"), 'pyquick')['energy']


@pytest.mark.parametrize("mode", MODES)
def test_same_energy_in_every_mode(tmp_path, serial_energy, mode):
    _need(mode, run=True)
    nranks = None
    if mode in ('mpi', 'gpu_mpi') and HAVE_MPIEXEC:
        nranks = 2 if mode == 'mpi' else min(2, NGPU)
    res = _energy(tmp_path, MODULES[mode], nranks=nranks)
    assert res['energy'] == pytest.approx(REF_HF_ENERGY, abs=TOL)
    assert res['energy'] == pytest.approx(serial_energy, abs=TOL_MODES)
    assert res['size'] == (nranks or 1)
