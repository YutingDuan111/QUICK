import atexit as _atexit
import copy as _copy
import os as _os

import numpy as _np

try:
    from ._pyquick import pyquick as _mod
except ImportError as e:
    if 'libquick' in str(e):
        raise ImportError(
            "QUICK shared library not found. "
            "Please source the QUICK environment setup script (quick.rc) "
            "before importing this module."
        ) from e
    raise


# ---------------------------------------------------------------------------
# MPI (an extension built with cmake -DMPI=TRUE, bound to libquick_mpi; with
# -DCUDA=TRUE too, libquick_mpi_cuda: one GPU per rank)
#
# Python owns MPI: importing mpi4py starts it, and mpi4py finalizes it at exit.
# QUICK borrows it in library mode (set_mpi), the way Amber does.  Every QUICK
# calculation runs on all ranks of MPI_COMM_WORLD; rank 0 reads the results
# from Fortran and broadcasts them, so every rank returns the same Result.
# Serial and GPU builds never import mpi4py.
# ---------------------------------------------------------------------------

_MPI_BUILD, _GPU_BUILD = (bool(flag) for flag in _mod.get_build_info())
_COMM = None
if _MPI_BUILD:
    try:
        from mpi4py import MPI as _MPI
    except ImportError as e:
        raise ImportError(
            "this pyquick was built with MPI and needs mpi4py, built against the "
            "same MPI library as QUICK:  MPICC=<path to mpicc> python -m pip "
            "install --no-binary=mpi4py mpi4py"
        ) from e
    _COMM = _MPI.COMM_WORLD
    _mod.set_mpi(_COMM.Get_rank(), _COMM.Get_size())


def mpi_enabled():
    """True if this pyquick was built with MPI (one calculation runs on all ranks)."""
    return _MPI_BUILD


def gpu_enabled():
    """True if this pyquick was built for GPUs (one GPU, or one per rank with MPI)."""
    return _GPU_BUILD


def mpi_rank():
    """This process's rank in MPI_COMM_WORLD (0 when not built with MPI)."""
    return _COMM.Get_rank() if _MPI_BUILD else 0


def mpi_size():
    """Number of MPI ranks a calculation runs on (1 when not built with MPI)."""
    return _COMM.Get_size() if _MPI_BUILD else 1


# QUICK's finalize is a collective call under MPI (it gathers timings from every
# rank), so it must run at the same point on all ranks: never from __del__, whose
# timing belongs to the garbage collector, and never after mpi4py has finalized
# MPI at exit.  Under MPI, __del__ only records that a cleanup is due; it runs at
# the next run()/job_open(), or in the exit hook below.
_cleanup_due = False


def _request_cleanup():
    global _cleanup_due
    _cleanup_due = True


def _run_due_cleanup():
    global _cleanup_due
    if _cleanup_due:
        _cleanup_due = False
        if _mod.job_active:
            _mod.job_destroy()


def _mpi_exit_hook():
    # registered after mpi4py started MPI, so it runs before mpi4py's own
    # finalization (exit hooks run last-in, first-out)
    if _mod.job_active:
        _mod.job_destroy()


if _MPI_BUILD:
    _atexit.register(_mpi_exit_hook)


def _checked(fn):
    """Wrap a Fortran subroutine call; raise RuntimeError if had_error is set."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if _mod.had_error:
            msg = bytes(_mod.error_message).decode().strip()
            _mod.had_error = False
            _mod.error_message = b' ' * 512
            raise RuntimeError(msg)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Low-level engine class
# ---------------------------------------------------------------------------

class _PyQuickEngine:
    """Object-oriented interface to a single QUICK calculation.

    Usage::

        job = _PyQuickEngine()
        job.set_calc('HF')
        job.set_basis('STO-3G')
        job.read_geom('''
            O  0.000  0.000  0.000
            H  0.757  0.586  0.000
            H -0.757  0.586  0.000
        ''')
        job.run()
        print(job.total_energy)
    """

    def __init__(self):
        self._calc    = None   # str, e.g. 'HF'
        self._basis   = None   # str, e.g. 'STO-3G'
        self._methods = []     # list of (keyword_str, arg_str_or_None)
        self._geom    = None   # raw geometry string as passed by user
        self._point_charges = None   # (n, 4) array of x, y, z (Angstrom), q; or None
        self._ran     = False
        self._results = {}     # snapshot of results captured at end of run()

    # -- setup methods -------------------------------------------------------

    def set_calc(self, keyword):
        """Set the calculation type: 'HF', 'UHF', 'DFT', or 'UDFT'."""
        _checked(_mod.set_calc)(keyword)
        self._calc = keyword

    def set_basis(self, basis_name):
        """Set the basis set, e.g. 'STO-3G', '6-31G*'."""
        _checked(_mod.set_basis)(basis_name)
        self._basis = basis_name

    def set_method(self, keyword, arg=None):
        """Add or update a keyword token in the job card.

        Examples::

            job.set_method('DIPOLE')          # bare keyword
            job.set_method('CUTOFF', '1e-9')  # keyword=value
        """
        if arg is not None:
            _checked(_mod.set_method)(keyword, arg)
        else:
            _checked(_mod.set_method)(keyword)
        uname = keyword.strip().upper()
        for i, (k, _) in enumerate(self._methods):
            if k == uname:
                self._methods[i] = (uname, arg)
                return
        self._methods.append((uname, arg))

    def unset_method(self, keyword):
        """Remove a keyword token previously added via set_method().

        Raises ValueError if the keyword is not currently set.
        """
        uname = keyword.strip().upper()
        for i, (k, _) in enumerate(self._methods):
            if k == uname:
                del self._methods[i]
                return
        raise ValueError(f"method keyword {keyword!r} is not set")

    def set_output(self, stem):
        """Set the file stem for QUICK's output (default ``'pyquick_job'``).

        QUICK writes its diagnostic log to ``<stem>.out`` unless the run is made
        with ``log=False``.  The stem also names auxiliary files QUICK writes
        when their keyword is set (e.g. ``<stem>.dat`` with ``CHK_WRITE``,
        ``<stem>.molden`` with ``MOLDEN``).
        """
        _checked(_mod.job_set_output)(stem)

    def read_geom(self, geom):
        """Set the molecular geometry.

        *geom* is a multi-line string with one atom per line::

            'SYMBOL  X  Y  Z'

        Coordinates are in Angstrom.
        """
        _checked(_mod.read_geom)(geom)
        self._geom = geom

    def set_point_charges(self, charges):
        """Set external point charges for the following runs, or clear them.

        *charges* is an ``(n, 4)`` float array of ``x, y, z`` (Angstrom) and
        charge, as returned by :func:`_check_point_charges`; ``None`` clears.
        """
        self._point_charges = None if charges is None else charges.copy()

    def _replay_point_charges(self):
        if self._point_charges is None:
            _mod.clear_point_charges()
        else:
            _checked(_mod.set_point_charges)(self._point_charges.T)

    def print_input(self):
        """Print the assembled QUICK input to stdout."""
        print(self.input_string)

    @property
    def input_string(self):
        """The assembled QUICK input as a string."""
        # replay this instance's state so Fortran's input_string reflects it
        _mod.clear_methods()
        if self._calc  is not None: _checked(_mod.set_calc)(self._calc)
        if self._basis is not None: _checked(_mod.set_basis)(self._basis)
        for keyword, arg in self._methods:
            if arg is not None: _checked(_mod.set_method)(keyword, arg)
            else:               _checked(_mod.set_method)(keyword)
        if self._geom  is not None: _checked(_mod.read_geom)(self._geom)
        return bytes(_mod.input_string).decode().strip()

    # -- execution -----------------------------------------------------------

    def run(self, jobtype=0, log=True, max_cycles=0):
        """Run a QUICK job on the current setup.

        *jobtype* selects the work: ``0`` = single-point energy, ``1`` = energy
        + nuclear gradient, ``2`` = geometry optimization.  *max_cycles* applies
        to ``jobtype=2`` only: ``<= 0`` uses the default cap of 2000 steps (QUICK's
        default), ``> 0`` caps the optimization cycles.  *log* writes QUICK's
        diagnostic output to
        ``<stem>.out`` (see :meth:`set_output`); ``log=False`` sends it to the
        platform null device so the run leaves no file behind.

        Must be called after :meth:`set_calc`, :meth:`set_basis`, and
        :meth:`read_geom`.  Results are snapshotted and available as properties
        afterwards.

        If this instance has been run before, the previous QUICK state is
        fully finalized before the new run begins, so basis sets and array
        dimensions are always consistent.  Successive runs append to the same
        log file; the first run replaces it.
        """
        self._results = {}
        self._ran = False
        if self._calc is None:
            raise RuntimeError("call set_calc() before run()")
        if self._basis is None:
            raise RuntimeError("call set_basis() before run()")
        if self._geom is None:
            raise RuntimeError("call read_geom() before run()")
        _run_due_cleanup()
        # replay this instance's state into the Fortran singleton
        _mod.clear_methods()
        _checked(_mod.set_calc)(self._calc)
        _checked(_mod.set_basis)(self._basis)
        for keyword, arg in self._methods:
            if arg is not None:
                _checked(_mod.set_method)(keyword, arg)
            else:
                _checked(_mod.set_method)(keyword)
        _checked(_mod.read_geom)(self._geom)
        self._replay_point_charges()
        _checked(_mod.job_run)(jobtype, 1 if log else 0, int(max_cycles))
        self._ran = True
        self._share_results()

    def _replay_setup(self):
        """Replay this instance's calc/basis/methods into the Fortran singleton."""
        _mod.clear_methods()
        _checked(_mod.set_calc)(self._calc)
        _checked(_mod.set_basis)(self._basis)
        for keyword, arg in self._methods:
            if arg is not None:
                _checked(_mod.set_method)(keyword, arg)
            else:
                _checked(_mod.set_method)(keyword)
        self._replay_point_charges()

    def job_open(self, log=True):
        """Set up a persistent job (allocate once) on the current setup."""
        if self._calc is None:
            raise RuntimeError("call set_calc() before job_open()")
        if self._basis is None:
            raise RuntimeError("call set_basis() before job_open()")
        if self._geom is None:
            raise RuntimeError("call read_geom() before job_open()")
        _run_due_cleanup()
        self._replay_setup()
        _checked(_mod.read_geom)(self._geom)
        _checked(_mod.job_open)(1 if log else 0)

    def job_step(self, geometry, gradient=False):
        """Run one geometry of the open job (reusing the density); snapshot results."""
        self._results = {}
        self._geom = geometry
        _checked(_mod.read_geom)(geometry)
        _checked(_mod.job_step)(1 if gradient else 0)
        self._ran = True
        self._share_results()

    def job_close(self):
        """Finalize the open job and free the allocated memory."""
        _checked(_mod.job_close)()

    def _share_results(self):
        """Snapshot the results; under MPI, rank 0 does it and broadcasts them.

        After a run only rank 0 holds complete results in Fortran (the others
        hold partial sums or nothing), so workers never read them.
        """
        if _MPI_BUILD:
            if _COMM.Get_rank() == 0:
                self._snapshot()
            self._results = _COMM.bcast(self._results, root=0)
        else:
            self._snapshot()

    def _snapshot(self):
        # snapshot all results into Python-owned storage so that a subsequent
        # run() on a different instance cannot overwrite this instance's results
        self._results['total_energy']      = float(_mod.job_total_energy)
        self._results['nuclear_repulsion'] = float(_mod.job_e_core)
        self._results['e_electronic']      = float(_mod.job_e_electronic)
        self._results['e_one_electron']    = float(_mod.job_e_1e)
        self._results['e_xc']              = float(_mod.job_e_xc)
        self._results['e_kinetic']         = float(_mod.job_e_kinetic)
        self._results['e_nuclear_attraction'] = (self._results['e_one_electron']
                                                 - self._results['e_kinetic'])
        # two-electron energy (J + K combined) by exact bookkeeping:
        # EEl = E1e + E2e + Exc  =>  E2e = EEl - E1e - Exc
        self._results['e_two_electron'] = (self._results['e_electronic']
                                           - self._results['e_one_electron']
                                           - self._results['e_xc'])
        # conditional energies: only present when their input keyword was active
        if _mod.job_has_dispersion:
            self._results['e_dispersion'] = float(_mod.job_e_disp)
        if _mod.job_has_extcharge:
            # ECharge is reported separately; it is never part of total_energy
            self._results['e_external_charge'] = float(_mod.job_e_charge)
        # geometry as parsed by QUICK itself (atomic numbers + Angstrom coords)
        atnums, coords, ngeom = _checked(_mod.job_get_geometry)()
        self._results['atomic_numbers'] = atnums[:ngeom].copy()
        self._results['coordinates'] = coords[:3 * ngeom].reshape(ngeom, 3).copy()
        if _mod.job_has_mulliken:
            r, n = _mod.job_get_mulliken()
            self._results['mulliken'] = r[:n].copy()
        if _mod.job_has_esp:
            r, n = _mod.job_get_esp_charges()
            self._results['esp_charges'] = r[:n].copy()
        if _mod.job_has_lowdin:
            r, n = _mod.job_get_lowdin()
            self._results['lowdin'] = r[:n].copy()
        if _mod.job_has_mo_energies:
            r, n = _mod.job_get_mo_energies()
            self._results['mo_energies'] = r[:n].copy()
        if _mod.job_has_density_matrix:
            r, nr, nc = _mod.job_get_density_matrix()
            self._results['density_matrix'] = r[:nr * nc].reshape(nr, nc).copy()
        nbasis, nmo = int(_mod.job_nbasis), int(_mod.job_nmo)
        self._results['mo_coefficients'] = _mod.job_get_mo_coefficients(
            nbasis * nmo).reshape(nbasis, nmo).copy()
        self._results['overlap_matrix'] = _mod.job_get_overlap(
            nbasis * nbasis).reshape(nbasis, nbasis).copy()
        self._results['ao_labels'] = _ao_labels(self._results['atomic_numbers'],
                                                *_mod.job_get_ao_info(nbasis))
        if _mod.job_has_beta:
            r, n = _mod.job_get_beta_mo_energies()
            self._results['beta_mo_energies'] = r[:n].copy()
            r, nr, nc = _mod.job_get_beta_density_matrix()
            self._results['beta_density_matrix'] = r[:nr * nc].reshape(nr, nc).copy()
            self._results['beta_mo_coefficients'] = _mod.job_get_beta_mo_coefficients(
                nbasis * nmo).reshape(nbasis, nmo).copy()
            self._results['mo_occupations'] = _occupations(nmo, int(_mod.job_nelec_alpha), 1.0)
            self._results['beta_mo_occupations'] = _occupations(nmo, int(_mod.job_nelec_beta), 1.0)
        else:
            self._results['mo_occupations'] = _occupations(nmo, int(_mod.job_nelec_alpha), 2.0)
        if _mod.job_has_dipole:
            self._results['dipole'] = _mod.job_dipole.copy()   # (3,) Debye
        if _mod.job_has_trajectory:
            nsteps = int(_mod.job_trajectory_nsteps)
            natoms = int(_mod.job_trajectory_natoms)
            self._results['trajectory_energies'] = _mod.job_get_trajectory_energies(nsteps).copy()
            self._results['trajectory_coordinates'] = _mod.job_get_trajectory_coordinates(
                nsteps * natoms * 3).reshape(nsteps, natoms, 3).copy()
        if _mod.job_has_ptchg_gradient:
            g, n = _mod.job_get_point_charge_gradient()
            self._results['point_charge_gradient'] = g[:3 * n].reshape(n, 3).copy()
        if _mod.job_has_gradient:
            g, n = _mod.job_get_gradients()
            self._results['gradient'] = g[:3 * n].reshape(n, 3).copy()
        if _mod.job_has_optimized:
            c, n = _mod.job_get_optimized_geometry()
            self._results['optimized_coordinates'] = c[:3 * n].reshape(n, 3).copy()
            self._results['converged'] = bool(_mod.job_opt_converged)

    def copy(self):
        """Return a new _PyQuickEngine with the same setup state.

        Results from a previous :meth:`run` are not copied — the new instance
        starts unrun.  All setup attributes (_calc, _basis, _methods, _geom)
        are independent copies, so changes to one object do not affect the other.
        """
        new = _PyQuickEngine()
        new._calc    = self._calc
        new._basis   = self._basis
        new._methods = list(self._methods)   # list of immutable tuples — shallow copy is sufficient
        new._geom    = self._geom
        new._point_charges = (None if self._point_charges is None
                              else self._point_charges.copy())
        return new

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        # setup state is strings and tuples of strings, so copy() is already
        # independent; never clone run state (__del__ would finalize QUICK)
        return self.copy()

    def __del__(self):
        # Only finalize QUICK if this object successfully ran a calculation.
        # Under MPI, defer it to a point every rank reaches (see _request_cleanup).
        try:
            if self._ran and _mod.job_active:
                if _MPI_BUILD:
                    _request_cleanup()
                else:
                    _mod.job_destroy()
        except Exception:
            pass

    # -- scalar results ------------------------------------------------------

    def _require_run(self, prop_name):
        if not self._ran:
            raise AttributeError(
                f"'{prop_name}' is not available until run() has been called"
            )

    @property
    def total_energy(self):
        """Total SCF energy in Hartree."""
        self._require_run('total_energy')
        return self._results['total_energy']

    @property
    def nuclear_repulsion(self):
        """Core-core nuclear repulsion energy in Hartree (QUICK's ECore)."""
        self._require_run('nuclear_repulsion')
        return self._results['nuclear_repulsion']

    @property
    def e_electronic(self):
        """Total electronic energy in Hartree."""
        self._require_run('e_electronic')
        return self._results['e_electronic']

    @property
    def e_one_electron(self):
        """One-electron energy (kinetic + nuclear-electron attraction) in Hartree."""
        self._require_run('e_one_electron')
        return self._results['e_one_electron']

    @property
    def e_two_electron(self):
        """Two-electron energy (Coulomb J + exchange K combined) in Hartree.

        Derived exactly as ``e_electronic - e_one_electron - e_xc``.
        """
        self._require_run('e_two_electron')
        return self._results['e_two_electron']

    @property
    def e_xc(self):
        """DFT exchange-correlation energy in Hartree (0.0 for pure HF)."""
        self._require_run('e_xc')
        return self._results['e_xc']

    @property
    def e_dispersion(self):
        """Empirical dispersion correction in Hartree.

        Only available when a dispersion keyword (D2/D3/D3BJ/...) was set.
        """
        self._require_run('e_dispersion')
        if 'e_dispersion' not in self._results:
            raise AttributeError(
                "'e_dispersion' is unavailable: no dispersion correction was "
                "requested for this job — set a dispersion keyword, e.g. "
                "set_method('D3')"
            )
        return self._results['e_dispersion']

    @property
    def e_external_charge(self):
        """Repulsion among the external point charges themselves, in Hartree.

        Only available when point charges were set. This term is reported
        separately and is never part of ``total_energy``.
        """
        self._require_run('e_external_charge')
        if 'e_external_charge' not in self._results:
            raise AttributeError(
                "'e_external_charge' is unavailable: no point charges were set "
                "for this job — set them via set_point_charges()"
            )
        return self._results['e_external_charge']

    # -- array results -------------------------------------------------------

    @property
    def mulliken(self):
        """Mulliken partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        self._require_run('mulliken')
        if 'mulliken' not in self._results:
            raise AttributeError(
                "'mulliken' charges were not computed; "
                "include DIPOLE in the keyword line via set_method('DIPOLE')"
            )
        return self._results['mulliken']

    @property
    def lowdin(self):
        """Lowdin partial charges as a numpy array of shape (natom,).

        Requires DIPOLE in the keyword line::

            job.set_method('DIPOLE')
        """
        self._require_run('lowdin')
        if 'lowdin' not in self._results:
            raise AttributeError(
                "'lowdin' charges were not computed; "
                "include DIPOLE in the keyword line via set_method('DIPOLE')"
            )
        return self._results['lowdin']

    @property
    def mo_energies(self):
        """Molecular orbital energies (alpha) as a numpy array of shape (NBSuse,)."""
        self._require_run('mo_energies')
        if 'mo_energies' not in self._results:
            raise AttributeError(
                "'mo_energies' were not computed; run() must complete successfully"
            )
        return self._results['mo_energies']

    @property
    def density_matrix(self):
        """Density matrix, numpy array of shape (nbasis, nbasis).

        The TOTAL density on restricted runs; the ALPHA density on unrestricted
        (open-shell) runs, where :attr:`beta_density_matrix` holds the beta part.
        """
        self._require_run('density_matrix')
        if 'density_matrix' not in self._results:
            raise AttributeError(
                "'density_matrix' was not computed; run() must complete successfully"
            )
        return self._results['density_matrix']

    @property
    def dipole(self):
        """Dipole moment vector (x, y, z) in Debye, shape (3,).

        Requires DIPOLE in the keyword line via set_method('DIPOLE').
        """
        self._require_run('dipole')
        if 'dipole' not in self._results:
            raise AttributeError(
                "'dipole' was not computed; include DIPOLE via set_method('DIPOLE')"
            )
        return self._results['dipole']

    @property
    def gradient(self):
        """Nuclear gradient dE/dR, shape (natom, 3), Hartree/Bohr.

        Requires a gradient run: ``run(jobtype=1)``.
        """
        self._require_run('gradient')
        if 'gradient' not in self._results:
            raise AttributeError(
                "'gradient' was not computed; run with run(jobtype=1)"
            )
        return self._results['gradient']


# ---------------------------------------------------------------------------
# High-level PyQuick / Result API
#
# Thin wrappers over the low-level _PyQuickEngine above.  The
# engine keeps the canonical pyquick conventions (check()-style errors, in-memory
# transfer, per-run snapshotting); PyQuick/Result are the documented surface.
# ---------------------------------------------------------------------------

# Log/output file stem used when a run is given no name. Mirrors the default of
# `output_stem` in pyquick.f90.
_DEFAULT_LOG_STEM = 'pyquick_job'


def read_xyz(path):
    """Read an .xyz file; returns its geometry as a 'SYMBOL X Y Z' string.

    The file holds the atom count, a comment line (may be empty), then one
    'SYMBOL X Y Z' line per atom in Angstrom; columns after Z are ignored.
    Raises ValueError for a malformed file or one with several geometries.
    """
    name = _os.fspath(path)
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        natoms = int(lines[0])
    except (IndexError, ValueError):
        natoms = 0
    if natoms < 1:
        raise ValueError(f"{name}: the first line must be the atom count")

    atom_lines = lines[2:2 + natoms]
    extra_lines = lines[2 + natoms:]
    found = sum(1 for line in atom_lines if line.strip())
    if found < natoms:
        raise ValueError(f"{name}: expected {natoms} atoms, found {found}")
    if any(line.strip() for line in extra_lines):
        raise ValueError(f"{name}: must hold one geometry, but there are lines "
                         f"after the {natoms} atoms")

    atoms = []
    for number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        try:
            [float(x) for x in fields[1:4]]
            if len(fields) < 4:
                raise ValueError
        except ValueError:
            raise ValueError(f"{name}: line {number}: expected SYMBOL X Y Z") from None
        atoms.append(' '.join(fields[:4]))
    return '\n'.join(atoms)


def _resolve_geometry(geometry):
    """Geometry string for *geometry*, reading the file if it is an .xyz path.

    A path is an os.PathLike or a string with no line break ending in .xyz;
    anything else is left for read_geom to parse.
    """
    if isinstance(geometry, _os.PathLike):
        return read_xyz(geometry)
    if isinstance(geometry, str):
        path = geometry.strip()
        if path.lower().endswith('.xyz') and '\n' not in path:
            return read_xyz(path)
    return geometry


def _count_atoms(geometry):
    """Number of atom lines ('SYMBOL X Y Z') in a geometry string.

    A light count for the geo_opt atom-count guard, not a full parse — QUICK
    still validates the geometry itself in read_geom.
    """
    return sum(1 for line in str(geometry).splitlines() if len(line.split()) >= 4)


def _occupations(nmo, noccupied, value):
    """Occupation numbers of *nmo* orbitals: the first *noccupied* hold *value*."""
    occupations = _np.zeros(nmo)
    occupations[:noccupied] = value
    return occupations


def _ao_labels(atomic_numbers, centers, powers):
    """(atom index, element, function) for each AO, e.g. ``(0, 'O', 'px')``."""
    labels = []
    for center, (lx, ly, lz) in zip(centers, powers.reshape(-1, 3)):
        atom = int(center) - 1
        function = 'spdfghi'[lx + ly + lz]
        if lx + ly + lz > 0:
            function += 'x' * lx + 'y' * ly + 'z' * lz
        labels.append((atom, _ELEMENTS[int(atomic_numbers[atom]) - 1], function))
    return labels


def _check_point_charges(point_charges):
    """Validate *point_charges*; return an ``(n, 4)`` float array, or None.

    Each row is ``(x, y, z, q)`` with coordinates in Angstrom.  ``None`` means
    no charges; an empty list is rejected rather than read as "no charges".
    """
    if point_charges is None:
        return None
    try:
        arr = _np.array(point_charges, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("point_charges must be numeric rows of (x, y, z, q)") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"point_charges must have shape (n, 4) -- rows of (x, y, z, q) in "
            f"Angstrom -- got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("point_charges is empty; pass None for no point charges")
    return arr

# The energy breakdown is ALWAYS returned (every job computes it), so it is not
# a "property" the caller opts into.
_ENERGY_ATTRS = ('total_energy', 'nuclear_repulsion', 'e_electronic',
                 'e_one_electron', 'e_two_electron', 'e_xc',
                 'e_kinetic', 'e_nuclear_attraction')

# Opt-in extras the caller selects via `properties`.  These cost extra work
# beyond the SCF (they need the DIPOLE keyword), so they are not returned unless
# asked for.
_SUPPORTED_PROPERTIES = {
    'mulliken_charges',  # needs the DIPOLE keyword to be computed
    'lowdin_charges',    # needs the DIPOLE keyword to be computed
    'dipole',            # dipole moment vector (Debye)
    'esp_charges',       # ESP-fitted atomic charges (ESP_CHARGE keyword)
}

# Always returned on the Result because every SCF computes them for free
# (no extra keyword, no extra routine).  Accepted in `properties` for backward
# compatibility, but they are a no-op there — you get them either way.
_ALWAYS_ON_PROPERTIES = {
    'mo_energies',       # alpha MO energies
    'density_matrix',    # total (restricted) or alpha (unrestricted) density
    'beta_mo_energies',      # unrestricted runs only
    'beta_density_matrix',   # unrestricted runs only
    'mo_coefficients',
    'mo_occupations',
    'overlap_matrix',
    'ao_labels',
    'beta_mo_coefficients',  # unrestricted runs only
    'beta_mo_occupations',   # unrestricted runs only
}

# These are all computed inside QUICK's `dipole` routine, which only runs when
# the DIPOLE keyword is set.
_NEEDS_DIPOLE = {'mulliken_charges', 'lowdin_charges', 'dipole'}

# Job-type words: not properties.  The job type is the method you call.
_JOBTYPE_WORDS = {'energy', 'gradient', 'gradients', 'forces',
                  'optimize', 'optimized_coords', 'optimized_coordinates', 'geo_opt'}

# Quantities QUICK can compute that pyquick deliberately does not expose --
# reported with a clearer message than "unknown property".
_UNSUPPORTED_PROPERTIES = {
    'mp2',     # MP2 energy: not part of pyquick (decided 2026-10-19)
    'e_mp2',
}

# opt-in property name -> Result attributes it unlocks.  The conditional energies
# (e_dispersion, e_external_charge) are gated by input keywords, not by the
# properties list — see Result.
_PROPERTY_ATTRS = {
    'mulliken_charges': ('mulliken',),
    'lowdin_charges': ('lowdin',),
    'dipole': ('dipole',),
    'esp_charges': ('esp_charges',),
}

# Raw QUICK keywords that pyquick replaces with its own API, or cannot support,
# mapped to the message telling the user what to do instead.  Passing them
# straight through used to run silently and return nothing.
_REJECTED_KEYWORDS = {
    'EXTCHARGES': "do not set the EXTCHARGES keyword; pass the charges per call "
                  "instead: get_energy(geometry, point_charges=[(x, y, z, q), ...])",
    'ESP_CHARGE': "do not set the ESP_CHARGE keyword; request "
                  "properties=['esp_charges'] instead",
    'ESP_GRID': "ESP_GRID (ESP on a user point file) is not supported by pyquick",
    'EFIELD_GRID': "EFIELD_GRID (field on a user point file) is not supported by pyquick",
    'EFG_GRID': "EFG_GRID (field gradient on a user point file) is not supported by pyquick",
}

# Element symbols by atomic number, for the electron count in _check_spin.
_ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni '
    'Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I '
    'Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt '
    'Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U').split()
_ATOMIC_NUMBER = {sym.upper(): z for z, sym in enumerate(_ELEMENTS, start=1)}


def _check_spin(geometry, charge, mult):
    """Raise ValueError if *charge* and *mult* are impossible for *geometry*.

    QUICK detects this too, but reports it by calling exit(), which kills the
    Python process (or Jupyter kernel) instead of raising.  Lines whose symbol
    is unknown are skipped; read_geom reports those itself.
    """
    nuclear = 0
    for line in str(geometry).splitlines():
        fields = line.split()
        if fields:
            nuclear += _ATOMIC_NUMBER.get(fields[0].upper(), 0)
    nelec = nuclear - charge
    unpaired = mult - 1
    if nelec < 0 or unpaired > nelec or (nelec - unpaired) % 2 != 0:
        raise ValueError(
            f"charge={charge} and mult={mult} are impossible for this molecule "
            f"({nelec} electrons): mult must be the number of unpaired "
            f"electrons + 1, so {nelec} electrons need an "
            f"{'even' if nelec % 2 else 'odd'} mult")


def _resolve_method(method, mult):
    """Map a user method + multiplicity to (calc_keyword, functional_token).

    'HF'/'RHF'/'UHF' are explicit Hartree-Fock; any other name is treated as a
    DFT functional and routed through DFT/UDFT.  mult > 1 selects the
    unrestricted path.
    """
    if not method or not str(method).strip():
        raise ValueError("PyQuick requires a 'method' (e.g. 'HF' or 'B3LYP')")
    if mult < 1:
        raise ValueError(f"multiplicity must be >= 1, got {mult}")

    m = str(method).strip().upper()
    open_shell = mult > 1

    if m in ('HF', 'RHF'):
        return ('UHF' if open_shell else 'HF'), None
    if m == 'UHF':
        return 'UHF', None
    if m in ('DFT', 'UDFT'):
        raise ValueError(
            "method='DFT' needs a functional name, e.g. method='B3LYP'")
    # Anything else is a DFT functional name.
    return ('UDFT' if open_shell else 'DFT'), m


class PyQuick:
    """A reusable set of QUICK calculation settings.

    The job type is chosen by which method you call:

        calc.get_energy(geometry)  -> single-point energy
        calc.get_grad(geometry)    -> energy + forces
        calc.geo_opt(geometry)     -> optimized geometry

    Example::

        calc = PyQuick(method='B3LYP', basis='6-31G*',
                       properties=['mulliken_charges'])
        result = calc.get_energy('''
            O  -0.338  0.004  0.239
            H  -0.335 -0.002 -0.833
            H   0.674 -0.002  0.594
        ''')
        print(result.total_energy)

    Parameters
    ----------
    method : str
        ``'HF'`` / ``'UHF'`` for Hartree-Fock, or a DFT functional name
        (``'B3LYP'``, ``'PBE0'``, ...) which routes through DFT/UDFT.
    basis : str
        Basis set name, e.g. ``'6-31G*'``.
    properties : iterable of str
        Opt-in extras to make available on the Result: ``'mulliken_charges'``,
        ``'lowdin_charges'``, ``'dipole'`` (all need the DIPOLE keyword).  The
        energy breakdown, geometry, ``mo_energies`` and ``density_matrix`` are
        **always** returned (every SCF computes them for free), so they are not
        listed here; naming ``mo_energies``/``density_matrix`` is accepted but
        has no effect.
    charge : int
        Total molecular charge (``CHARGE=``). Default 0.
    mult : int
        Spin multiplicity (``MULT=``); ``mult > 1`` selects the unrestricted path.
    keywords : dict
        Extra QUICK keyword tokens. ``{'cutoff': '1e-9'}`` -> ``CUTOFF=1e-9``;
        a value of ``None`` emits a bare flag.
    log : bool
        Write QUICK's diagnostic log (default ``True``).  The file is named after
        the run's *name* -- ``get_energy(geom, name='water')`` -> ``water.out`` --
        or ``pyquick_job.out`` when no name is given.  Successive runs of the same
        PyQuick object append; an existing file is backed up to ``<file>~``.
        ``log=False`` sends the log to the platform null device, so the run leaves
        nothing on disk.
    """

    def __init__(self, method, basis, properties=(),
                 charge=0, mult=1, keywords=None, log=True):
        if not basis or not str(basis).strip():
            raise ValueError("PyQuick requires a 'basis' (e.g. '6-31G*')")

        requested = {str(p).strip().lower() for p in properties}
        for p in requested:
            if p in _SUPPORTED_PROPERTIES:
                continue
            if p in _ALWAYS_ON_PROPERTIES:
                continue                       # always returned; naming it is a no-op
            if p in _JOBTYPE_WORDS:
                raise ValueError(
                    f"{p!r} is not a property — the job type is chosen by the "
                    f"method you call: get_energy(), get_grad(), or geo_opt(). "
                    f"`properties` selects only extras: {sorted(_SUPPORTED_PROPERTIES)}")
            if p in _UNSUPPORTED_PROPERTIES:
                raise ValueError(
                    f"property {p!r} is not supported by pyquick; "
                    f"supported: {sorted(_SUPPORTED_PROPERTIES)}")
            raise ValueError(
                f"unknown property {p!r}; supported: {sorted(_SUPPORTED_PROPERTIES)}")

        self.method = method
        self.basis = basis
        self.properties = requested
        self.charge = int(charge)
        self.mult = int(mult)
        self.keywords = dict(keywords) if keywords else {}
        for k in self.keywords:
            message = _REJECTED_KEYWORDS.get(str(k).strip().upper())
            if message:
                raise ValueError(message)
        self.log = bool(log)
        self._calc_keyword, self._functional = _resolve_method(method, self.mult)
        self._engine = None

    def _build_engine(self):
        job = _PyQuickEngine()
        job.set_calc(self._calc_keyword)
        job.set_basis(self.basis)
        if self._functional is not None:
            job.set_method(self._functional)
        if any(p in _NEEDS_DIPOLE for p in self.properties):
            job.set_method('DIPOLE')
        if 'esp_charges' in self.properties:
            job.set_method('ESP_CHARGE')
        if self.charge != 0:
            job.set_method('CHARGE', str(self.charge))
        if self.mult != 1:
            job.set_method('MULT', str(self.mult))
        for key, value in self.keywords.items():
            if value is None:
                job.set_method(key)
            else:
                job.set_method(key, str(value))
        return job

    def _prepare(self, geometry, name, point_charges):
        """Shared setup for a run: engine, output stem, geometry, point charges."""
        _check_spin(geometry, self.charge, self.mult)
        if self._engine is None:
            self._engine = self._build_engine()
        job = self._engine
        # Set the stem on every run, not just when a name is given: it is Fortran
        # module state that would otherwise persist from an earlier run (or from
        # another PyQuick object), silently sending this run's log to that file.
        job.set_output(str(name) if name is not None else _DEFAULT_LOG_STEM)
        job.read_geom(geometry)
        # always set (or clear): charges belong to this call only
        job.set_point_charges(_check_point_charges(point_charges))
        return job

    def get_energy(self, geometry, name=None, point_charges=None):
        """Single-point energy on *geometry*; returns a :class:`Result`.

        *geometry* is a 'SYMBOL X Y Z' multi-line string (Angstrom) or the path of
        an .xyz file (see :func:`read_xyz`).  *name*
        labels the run (it lands in ``Result.metadata['name']``) and, when
        logging is on, names the log file — ``name='water'`` -> ``water.out``.
        A single engine is reused across successive calls on the same
        PyQuick object, so each call fully finalizes the previous one inside QUICK.

        *point_charges* adds external point charges for this call only: rows of
        ``(x, y, z, q)``, coordinates in Angstrom.  They act on the molecule
        (density, ``nuclear_repulsion``, gradient); their mutual repulsion is
        reported as ``e_external_charge`` and is never part of ``total_energy``.
        """
        geometry = _resolve_geometry(geometry)
        job = self._prepare(geometry, name, point_charges)
        job.run(jobtype=0, log=self.log)
        return Result._from_engine(job, self, geometry, name)

    def get_grad(self, geometry, name=None, point_charges=None):
        """Energy + nuclear gradient on *geometry*; returns a :class:`Result`.

        The gradient job runs the SCF and the gradient together, so the returned
        Result carries **both** the energy breakdown and ``gradient`` (dE/dR,
        shape ``(natom, 3)``, Hartree/Bohr) — the energy is not computed twice.
        *name* labels the run and, when logging is on, names the log file.
        *point_charges* is as for :meth:`get_energy`.
        """
        geometry = _resolve_geometry(geometry)
        job = self._prepare(geometry, name, point_charges)
        job.run(jobtype=1, log=self.log)
        return Result._from_engine(job, self, geometry, name)

    def geo_opt(self, geometry, name=None, max_cycles=None, point_charges=None):
        """Optimize *geometry*; returns a :class:`Result`.

        Starting from *geometry*, QUICK relaxes the structure and the Result
        carries ``optimized_coordinates`` (final structure, Angstrom),
        ``converged`` (did it reach the convergence criteria, or just run out of
        cycles?), and the energy breakdown **of the optimized structure**.
        ``coordinates`` still holds the geometry you passed in.

        *max_cycles* caps the optimization; the default (``None``) is 2000 steps.

        The Result also carries the path: ``trajectory_energies`` and
        ``trajectory_coordinates``, one entry per step from the input geometry to
        the final one.

        The optimizer is QUICK's default (DL-Find) unless you add the Cartesian
        optimizer via ``keywords={'LOPT': None}``.  DL-Find does not support
        molecules with fewer than 3 atoms — for those, use ``LOPT`` (this raises
        a ``ValueError`` otherwise).

        Always check ``result.converged`` — a run that hits *max_cycles* returns
        the last step's geometry, which is **not** optimized.

        *point_charges* is as for :meth:`get_energy`; the charges stay fixed
        while the molecule relaxes.
        """
        geometry = _resolve_geometry(geometry)
        # DL-Find (the default) fails on < 3 atoms; steer the user to LOPT.
        using_lopt = any(str(k).upper() == 'LOPT' for k in self.keywords)
        if not using_lopt and _count_atoms(geometry) < 3:
            raise ValueError(
                "DL-Find (the default geometry optimizer) does not support "
                "molecules with fewer than 3 atoms. Use the Cartesian optimizer: "
                "PyQuick(..., keywords={'LOPT': None}).")

        job = self._prepare(geometry, name, point_charges)
        job.run(jobtype=2, log=self.log,
                max_cycles=0 if max_cycles is None else int(max_cycles))
        return Result._from_engine(job, self, geometry, name)

    def new_job(self, geometry, reuse_density=True, point_charges=None):
        """Open a persistent :class:`Job` for one molecule.

        Set up the molecule and settings once, then call ``job.run(geom)`` for
        successive geometries of the **same** molecule (same atoms; only the
        coordinates change).  Each step reuses the converged density from the
        previous one as its SCF guess, so later steps converge faster.  Call
        ``job.delete()`` (or use the job as a context manager) to free the memory.

        *point_charges* (as for :meth:`get_energy`) is fixed for the whole job.
        """
        return Job(self, geometry, reuse_density=reuse_density,
                   point_charges=point_charges)

    def copy(self):
        """Return a shallow copy, like :func:`copy.copy`.

        Mutable settings (``keywords``, ``properties``) are shared with the
        original.  The copy always gets its own engine, built on its first run,
        so later edits to the copy's settings take effect.
        """
        return _copy.copy(self)

    def deepcopy(self):
        """Return a fully independent copy, like :func:`copy.deepcopy`."""
        return _copy.deepcopy(self)

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new._engine = None
        return new

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = None if key == '_engine' else _copy.deepcopy(value, memo)
        return new

    @property
    def input_string(self):
        """The assembled QUICK keyword line + basis (no geometry until run)."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine.input_string


_BETA_ONLY = ("exists only for unrestricted (open-shell) runs; on a restricted run "
              "alpha and beta are identical")


class Result:
    """Results of a completed :class:`PyQuick` run.

    Properties are accessed as attributes.  Accessing one that was not requested
    in ``PyQuick.properties`` (or could not be computed) raises
    ``AttributeError``.
    """

    def __init__(self, values, requested, metadata):
        self._values = dict(values)
        self._requested = set(requested)
        self.metadata = dict(metadata)

    def copy(self):
        """Return a shallow copy, like :func:`copy.copy` (arrays are shared)."""
        return _copy.copy(self)

    def deepcopy(self):
        """Return a fully independent copy, like :func:`copy.deepcopy`."""
        return _copy.deepcopy(self)

    @classmethod
    def _from_engine(cls, job, calc, geometry, name):
        values = {}
        for attr in _ENERGY_ATTRS:
            values[attr] = job._results[attr]
        # conditional energies: present only if their input keyword was active
        for key in ('e_dispersion', 'e_external_charge'):
            if key in job._results:
                values[key] = job._results[key]
        # geometry as parsed by QUICK (atomic numbers + Angstrom coordinates)
        values['atomic_numbers'] = job._results['atomic_numbers']
        values['coordinates'] = job._results['coordinates']
        for key in ('mulliken', 'lowdin', 'esp_charges', 'mo_energies', 'density_matrix',
                    'beta_mo_energies', 'beta_density_matrix',
                    'mo_coefficients', 'mo_occupations', 'overlap_matrix', 'ao_labels',
                    'beta_mo_coefficients', 'beta_mo_occupations',
                    'dipole', 'gradient', 'point_charge_gradient',
                    'optimized_coordinates', 'converged',
                    'trajectory_energies', 'trajectory_coordinates'):
            if key in job._results:
                values[key] = job._results[key]
        metadata = {
            'method': calc.method,
            'calc_keyword': calc._calc_keyword,
            'functional': calc._functional,
            'basis': calc.basis,
            'charge': calc.charge,
            'mult': calc.mult,
            'name': name,
            'geometry': geometry,
        }
        return cls(values, calc.properties, metadata)

    def _get(self, prop, attr):
        if prop not in self._requested:
            raise AttributeError(
                f"'{attr}' is unavailable: '{prop}' was not requested for this "
                f"calculation — add '{prop}' to PyQuick.properties")
        if attr not in self._values:
            raise AttributeError(
                f"'{attr}' was requested but not computed by QUICK for this job")
        return self._values[attr]

    # -- energy (always available) ------------------------------------------
    @property
    def total_energy(self):
        """Total energy in Hartree.

        ``total_energy = nuclear_repulsion + e_electronic (+ e_dispersion if
        a dispersion keyword was requested)``.
        """
        return self._values['total_energy']

    @property
    def nuclear_repulsion(self):
        """Core-core nuclear repulsion energy in Hartree."""
        return self._values['nuclear_repulsion']

    @property
    def e_electronic(self):
        """Total electronic energy in Hartree.

        ``e_electronic = e_one_electron + e_two_electron + e_xc``.
        """
        return self._values['e_electronic']

    @property
    def e_one_electron(self):
        """One-electron energy in Hartree: ``e_kinetic + e_nuclear_attraction``."""
        return self._values['e_one_electron']

    @property
    def e_kinetic(self):
        """Electronic kinetic energy ``Tr(P*T)`` in Hartree."""
        return self._values['e_kinetic']

    @property
    def e_nuclear_attraction(self):
        """Attraction of the electrons to the nuclei in Hartree.

        Equals ``e_one_electron - e_kinetic`` exactly.  With ``point_charges`` it
        also holds the electrons' attraction to those charges.
        """
        return self._values['e_nuclear_attraction']

    @property
    def e_two_electron(self):
        """Two-electron energy (Coulomb J + exchange K combined) in Hartree."""
        return self._values['e_two_electron']

    @property
    def e_xc(self):
        """DFT exchange-correlation energy in Hartree (0.0 for pure HF)."""
        return self._values['e_xc']

    # -- conditional energies (gated by input keywords, not `properties`) -----
    @property
    def e_dispersion(self):
        """Empirical dispersion correction in Hartree.

        Only available when a dispersion keyword was set; when present it is
        included in ``total_energy``.
        """
        if 'e_dispersion' not in self._values:
            raise AttributeError(
                "'e_dispersion' is unavailable: no dispersion correction was "
                "requested for this calculation — add a dispersion keyword, "
                "e.g. PyQuick(..., keywords={'D3': None})")
        return self._values['e_dispersion']

    @property
    def e_external_charge(self):
        """Repulsion among the external point charges themselves, in Hartree.

        Only available when the run was given ``point_charges``.  Reported
        separately; never included in ``total_energy``.
        """
        if 'e_external_charge' not in self._values:
            raise AttributeError(
                "'e_external_charge' is unavailable: no point charges were "
                "given for this calculation — pass them via "
                "get_energy(geometry, point_charges=[(x, y, z, q), ...])")
        return self._values['e_external_charge']

    # -- always available (every SCF computes them) --------------------------
    @property
    def mo_energies(self):
        """Molecular orbital energies, numpy array of shape (NBSuse,).

        The alpha orbitals on unrestricted (open-shell) runs; see
        :attr:`beta_mo_energies` for the beta ones.
        """
        if 'mo_energies' not in self._values:
            raise AttributeError("'mo_energies' was not computed for this run")
        return self._values['mo_energies']

    @property
    def density_matrix(self):
        """Density matrix, numpy array of shape (nbasis, nbasis).

        The TOTAL density on restricted runs (RHF, closed-shell DFT); the ALPHA
        density on unrestricted (open-shell) runs, where
        :attr:`beta_density_matrix` holds the beta part (total = alpha + beta).
        """
        if 'density_matrix' not in self._values:
            raise AttributeError("'density_matrix' was not computed for this run")
        return self._values['density_matrix']

    @property
    def beta_mo_energies(self):
        """Beta molecular orbital energies, numpy array of shape (NBSuse,).

        Unrestricted (open-shell) runs only.
        """
        if 'beta_mo_energies' not in self._values:
            raise AttributeError("'beta_mo_energies' exists only for unrestricted (open-shell) runs; on a restricted run alpha and beta are identical")
        return self._values['beta_mo_energies']

    @property
    def beta_density_matrix(self):
        """Beta density matrix, numpy array of shape (nbasis, nbasis).

        Unrestricted (open-shell) runs only; :attr:`density_matrix` is then
        the alpha density.
        """
        if 'beta_density_matrix' not in self._values:
            raise AttributeError("'beta_density_matrix' exists only for unrestricted (open-shell) runs; on a restricted run alpha and beta are identical")
        return self._values['beta_density_matrix']

    @property
    def mo_coefficients(self):
        """AO -> MO coefficients, numpy array of shape (nbasis, nmo); columns are MOs.

        The alpha orbitals on unrestricted runs, ordered as :attr:`mo_energies`,
        with rows ordered as :attr:`ao_labels`.  Orbital signs are arbitrary.
        """
        return self._values['mo_coefficients']

    @property
    def mo_occupations(self):
        """Occupation of each MO, shape (nmo,): 2/0 on restricted runs, else 1/0."""
        return self._values['mo_occupations']

    @property
    def overlap_matrix(self):
        """AO overlap matrix, numpy array of shape (nbasis, nbasis)."""
        return self._values['overlap_matrix']

    @property
    def ao_labels(self):
        """(atom index, element, function) for each AO, e.g. ``(0, 'O', 'px')``.

        Atom indices count from 0.  Functions are Cartesian (six d, ten f):
        ``'s'``, ``'px'``, ``'dxy'``, ...  This is the row order of
        :attr:`mo_coefficients` and of the density and overlap matrices.
        """
        return self._values['ao_labels']

    @property
    def beta_mo_coefficients(self):
        """Beta AO -> MO coefficients, shape (nbasis, nmo); unrestricted runs only."""
        if 'beta_mo_coefficients' not in self._values:
            raise AttributeError("'beta_mo_coefficients' %s" % _BETA_ONLY)
        return self._values['beta_mo_coefficients']

    @property
    def beta_mo_occupations(self):
        """Occupation of each beta MO, shape (nmo,); unrestricted runs only."""
        if 'beta_mo_occupations' not in self._values:
            raise AttributeError("'beta_mo_occupations' %s" % _BETA_ONLY)
        return self._values['beta_mo_occupations']

    # -- opt-in arrays (need the DIPOLE keyword) -----------------------------
    @property
    def mulliken(self):
        """Mulliken partial charges, numpy array of shape (natom,)."""
        return self._get('mulliken_charges', 'mulliken')

    @property
    def lowdin(self):
        """Lowdin partial charges, numpy array of shape (natom,)."""
        return self._get('lowdin_charges', 'lowdin')

    @property
    def dipole(self):
        """Dipole moment vector (x, y, z) in Debye, numpy array of shape (3,)."""
        return self._get('dipole', 'dipole')

    @property
    def esp_charges(self):
        """ESP-fitted atomic charges, numpy array of shape (natom,).

        Fitted to the electrostatic potential on Merz-Kollman-Singh surfaces;
        they sum to the molecular charge.  Request ``properties=['esp_charges']``.
        """
        return self._get('esp_charges', 'esp_charges')

    # -- gradient (produced by get_grad, not by a `properties` request) -------
    @property
    def gradient(self):
        """Nuclear gradient dE/dR, numpy array of shape (natom, 3), Hartree/Bohr.

        Only present on a Result from :meth:`PyQuick.get_grad`.
        """
        if 'gradient' not in self._values:
            raise AttributeError(
                "'gradient' is unavailable: this Result came from get_energy() — "
                "use calc.get_grad(geometry) to compute the nuclear gradient")
        return self._values['gradient']

    @property
    def point_charge_gradient(self):
        """Gradient on the external point charges, shape (ncharges, 3), Hartree/Bohr.

        dE/dR of each charge from :meth:`PyQuick.get_grad` with ``point_charges``
        (or ``job.run(geometry, gradient=True)``).  Like ``total_energy`` it leaves
        out the repulsion among the charges themselves.
        """
        if 'point_charge_gradient' not in self._values:
            if 'e_external_charge' not in self._values:
                raise AttributeError(
                    "'point_charge_gradient' is unavailable: this run had no "
                    "point_charges")
            raise AttributeError(
                "'point_charge_gradient' is unavailable: this run computed no "
                "gradient — use calc.get_grad(geometry, point_charges=...)")
        return self._values['point_charge_gradient']

    # -- optimization (produced by geo_opt) ----------------------------------
    @property
    def trajectory_energies(self):
        """Energy of each optimization step, shape (nsteps,), Hartree.

        Entry 0 is the input geometry, the last is the final one.  Only present on
        a Result from :meth:`PyQuick.geo_opt`.  QUICK records ``Eel + Ecore`` per
        step, so these energies leave out any dispersion correction.
        """
        if 'trajectory_energies' not in self._values:
            raise AttributeError(
                "'trajectory_energies' is unavailable: only geo_opt() produces a "
                "trajectory")
        return self._values['trajectory_energies']

    @property
    def trajectory_coordinates(self):
        """Geometry of each optimization step, shape (nsteps, natom, 3), Angstrom.

        Entry 0 is the input geometry, the last is the final one.  Only present on
        a Result from :meth:`PyQuick.geo_opt`.
        """
        if 'trajectory_coordinates' not in self._values:
            raise AttributeError(
                "'trajectory_coordinates' is unavailable: only geo_opt() produces a "
                "trajectory")
        return self._values['trajectory_coordinates']

    @property
    def optimized_coordinates(self):
        """Optimized atomic coordinates in Angstrom, shape (natom, 3).

        Only present on a Result from :meth:`PyQuick.geo_opt`.  Check
        :attr:`converged` before trusting it — a run that hit the cycle limit
        returns the last step's geometry, not an optimized one.
        """
        if 'optimized_coordinates' not in self._values:
            raise AttributeError(
                "'optimized_coordinates' is unavailable: this Result did not come "
                "from an optimization — use calc.geo_opt(geometry)")
        return self._values['optimized_coordinates']

    @property
    def converged(self):
        """True if the geometry optimization reached its convergence criteria.

        False means it stopped at the cycle limit, so ``optimized_coordinates``
        is the last step's geometry and is **not** optimized.
        """
        if 'converged' not in self._values:
            raise AttributeError(
                "'converged' is unavailable: it only applies to an optimization — "
                "use calc.geo_opt(geometry)")
        return self._values['converged']

    # -- geometry (always available, as parsed by QUICK) ---------------------
    @property
    def atomic_numbers(self):
        """Atomic numbers as a numpy int array of shape (natom,)."""
        return self._values['atomic_numbers']

    @property
    def coordinates(self):
        """Atomic coordinates in Angstrom, numpy array of shape (natom, 3)."""
        return self._values['coordinates']


class Job:
    """A persistent QUICK job for one molecule (see :meth:`PyQuick.new_job`).

    The molecule and settings are set up once; ``run(geometry)`` then computes
    successive geometries of the **same** molecule (same atoms, moving
    coordinates), reusing the converged density from the previous step as the
    next SCF's initial guess.  Free the allocation with :meth:`delete` or by
    using the job as a context manager::

        with calc.new_job(geom0) as job:
            energies = [job.run(g).total_energy for g in geometries]
    """

    def __init__(self, calc, geometry, reuse_density=True, point_charges=None):
        geometry = _resolve_geometry(geometry)
        self._calc = calc
        self._reuse_density = bool(reuse_density)   # reserved; density is reused
        self._natoms = _count_atoms(geometry)
        _check_spin(geometry, calc.charge, calc.mult)
        self._engine = calc._build_engine()
        self._engine.read_geom(geometry)            # fixes the molecule (atoms)
        self._engine.set_point_charges(_check_point_charges(point_charges))
        self._engine.job_open(log=calc.log)
        self._open = True

    def run(self, geometry, gradient=False):
        """Compute *geometry* (same molecule) and return a :class:`Result`.

        With ``gradient=True`` the nuclear gradient is computed for this step too.
        """
        if not self._open:
            raise RuntimeError("this Job has been deleted")
        geometry = _resolve_geometry(geometry)
        if _count_atoms(geometry) != self._natoms:
            raise ValueError(
                "a Job is fixed to one molecule; the atom count changed — "
                "open a new job for a different molecule")
        self._engine.job_step(geometry, gradient=gradient)
        return Result._from_engine(self._engine, self._calc, geometry, name=None)

    def delete(self):
        """Finalize the job and free its allocated memory."""
        if self._open:
            self._engine.job_close()
            self._open = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.delete()

    def __copy__(self):
        raise TypeError("a Job owns live QUICK memory and cannot be copied; "
                        "open another with PyQuick.new_job()")

    def __deepcopy__(self, memo):
        self.__copy__()

    def __del__(self):
        try:
            if _MPI_BUILD and self._open:
                # QUICK's finalize is collective: defer it (see _request_cleanup)
                self._open = False
                _request_cleanup()
            else:
                self.delete()
        except Exception:
            pass
