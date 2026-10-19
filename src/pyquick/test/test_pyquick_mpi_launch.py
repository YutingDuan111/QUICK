"""
MPI launcher tests for pyquick.  Run with PLAIN pytest -- each test starts its
own `mpiexec` in a subprocess:

    python -m pytest test_pyquick_mpi_launch.py -q

Requirements:
  * the pyquick that `python` imports is the MPI build (cmake -DMPI=TRUE);
    otherwise the whole file is skipped;
  * `mpiexec` on PATH.

Optional environment variables:
  PYQUICK_MPIEXEC     launcher command, default "mpiexec".  On a machine with
                      fewer cores than ranks, Open MPI needs
                      "mpiexec --oversubscribe".
  PYQUICK_SERIAL_LIB  the `lib` directory of a SERIAL pyquick install; enables
                      test_task_farm_example (task farming needs the serial build).

Each case runs in a fresh process, which also matches how the `quick` binary
references were made (one calculation per process).
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MPIEXEC = shlex.split(os.environ.get("PYQUICK_MPIEXEC", "mpiexec"))
TIMEOUT = 300                       # seconds for one launched job


def _probe(pythonpath=None):
    """Ask a fresh interpreter whether its pyquick is the MPI build (None = no pyquick)."""
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    code = "import pyquick; print(getattr(pyquick, 'mpi_enabled', lambda: False)())"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, timeout=120)
    return None if p.returncode != 0 else p.stdout.strip().endswith("True")


if shutil.which(MPIEXEC[0]) is None:
    pytest.skip(f"{MPIEXEC[0]} not on PATH", allow_module_level=True)
if not _probe():
    pytest.skip("the pyquick on PYTHONPATH is not the MPI build", allow_module_level=True)


WATER = "O -0.33840 0.00380 0.23923\nH -0.33510 -0.00190 -0.83277\nH 0.67350 -0.00190 0.59353"

ENERGY_SCRIPT = textwrap.dedent(f"""
    import json, pyquick
    out = {{}}
    for method in ('HF', 'B3LYP'):
        calc = pyquick.PyQuick(method, '6-31G*', log=False,
                               keywords={{'cutoff': '1.0e-11', 'denserms': '1.0e-8'}})
        out[method] = calc.get_energy({WATER!r}).total_energy
    out['size'] = pyquick.mpi_size()
    if pyquick.mpi_rank() == 0:
        print('RESULT ' + json.dumps(out))
""")


def _launch(tmp_path, script, nranks=None, env_extra=None):
    """Run `script` under mpiexec -n nranks (or plain python if nranks is None)."""
    path = tmp_path / "job.py"
    path.write_text(script)
    if nranks is None:
        cmd = [sys.executable, str(path)]
    else:
        cmd = MPIEXEC + ["-n", str(nranks), sys.executable, "-m", "mpi4py", str(path)]
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path,
                          env=env, timeout=TIMEOUT)


def _result(proc):
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert len(lines) == 1, f"expected one RESULT line (rank 0 only), got {lines}"
    return json.loads(lines[0][len("RESULT "):])


@pytest.fixture(scope="module")
def one_rank(tmp_path_factory):
    return _result(_launch(tmp_path_factory.mktemp("n1"), ENERGY_SCRIPT, nranks=1))


@pytest.mark.parametrize("nranks", [2, 3, 4])
def test_rank_count_does_not_change_energy(tmp_path, one_rank, nranks):
    """Only the summation order changes with the rank count."""
    if nranks > (os.cpu_count() or 1) and "PYQUICK_MPIEXEC" not in os.environ:
        pytest.skip(f"{nranks} ranks > {os.cpu_count()} cores; set PYQUICK_MPIEXEC to oversubscribe")
    res = _result(_launch(tmp_path, ENERGY_SCRIPT, nranks=nranks))
    assert res["size"] == nranks
    for method in ("HF", "B3LYP"):
        assert res[method] == pytest.approx(one_rank[method], abs=1e-8), method


def test_singleton_without_mpiexec(tmp_path, one_rank):
    """`python script.py` with the MPI build runs as a single rank."""
    res = _result(_launch(tmp_path, ENERGY_SCRIPT, nranks=None))
    assert res["size"] == 1
    for method in ("HF", "B3LYP"):
        assert res[method] == pytest.approx(one_rank[method], abs=1e-10), method


def test_engine_error_aborts_without_hanging(tmp_path):
    """An error inside QUICK (unknown basis) stops every rank: non-zero exit, no hang.

    subprocess's timeout turns a hang into a test failure instead of a stuck run.
    """
    script = textwrap.dedent(f"""
        import pyquick
        r = pyquick.PyQuick('HF', 'NOT-A-REAL-BASIS', log=False).get_energy({WATER!r})
        print('RESULT unexpectedly finished', r.total_energy)
    """)
    try:
        proc = _launch(tmp_path, script, nranks=2)
    except subprocess.TimeoutExpired:
        pytest.fail(f"job hung for {TIMEOUT} s instead of aborting")
    assert proc.returncode != 0
    assert "RESULT" not in proc.stdout


def test_task_farm_example(tmp_path):
    """The build guide's task-farming script (serial build) under mpiexec -n 2."""
    serial_lib = os.environ.get("PYQUICK_SERIAL_LIB")
    if not serial_lib:
        pytest.skip("set PYQUICK_SERIAL_LIB to a serial pyquick install's lib directory")
    assert _probe(serial_lib) is False, "PYQUICK_SERIAL_LIB must point at a SERIAL build"
    env = {"PYTHONPATH": serial_lib + os.pathsep + os.environ.get("PYTHONPATH", "")}

    farm = open(os.path.join(HERE, "mpi_task_farm_example.py")).read()
    proc = _launch(tmp_path, farm, nranks=2, env_extra=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    farmed = dict(l.split()[1:] for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    assert sorted(farmed) == sorted(["STO-3G", "3-21G", "6-31G", "6-31G*"])

    # same numbers as running each basis one by one in a single serial process
    check = textwrap.dedent("""
        import pyquick
        W = "O 0.0 0.0 0.0\\nH 0.0 0.0 0.96\\nH 0.93 0.0 -0.24"
        for b in ("STO-3G", "3-21G", "6-31G", "6-31G*"):
            e = pyquick.PyQuick(method="HF", basis=b, log=False).get_energy(W).total_energy
            print(f"RESULT {b} {e:.10f}")
    """)
    ref = _launch(tmp_path, check, nranks=None, env_extra=env)
    assert ref.returncode == 0, ref.stderr[-2000:]
    serial = dict(l.split()[1:] for l in ref.stdout.splitlines() if l.startswith("RESULT "))
    for basis, energy in serial.items():
        assert float(farmed[basis]) == pytest.approx(float(energy), abs=1e-8), basis
