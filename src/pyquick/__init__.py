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
# PyQuick class
# ---------------------------------------------------------------------------

class PyQuick:
    """Object-oriented interface to a single QUICK calculation.

    Usage::

        job = PyQuick()
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
        to ``jobtype=2`` only: ``<= 0`` runs to convergence with no cap (QUICK's
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
        _checked(_mod.job_run)(jobtype, 1 if log else 0, int(max_cycles))
        self._ran = True
        # snapshot all results into Python-owned storage so that a subsequent
        # run() on a different instance cannot overwrite this instance's results
        self._results['total_energy']      = float(_mod.job_total_energy)
        self._results['nuclear_repulsion'] = float(_mod.job_e_core)
        self._results['e_electronic']      = float(_mod.job_e_electronic)
        self._results['e_one_electron']    = float(_mod.job_e_1e)
        self._results['e_xc']              = float(_mod.job_e_xc)
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
        if _mod.job_has_lowdin:
            r, n = _mod.job_get_lowdin()
            self._results['lowdin'] = r[:n].copy()
        if _mod.job_has_mo_energies:
            r, n = _mod.job_get_mo_energies()
            self._results['mo_energies'] = r[:n].copy()
        if _mod.job_has_density_matrix:
            r, nr, nc = _mod.job_get_density_matrix()
            self._results['density_matrix'] = r[:nr * nc].reshape(nr, nc).copy()
        if _mod.job_has_dipole:
            self._results['dipole'] = _mod.job_dipole.copy()   # (3,) Debye
        if _mod.job_has_gradient:
            g, n = _mod.job_get_gradients()
            self._results['gradient'] = g[:3 * n].reshape(n, 3).copy()
        if _mod.job_has_optimized:
            c, n = _mod.job_get_optimized_geometry()
            self._results['optimized_coordinates'] = c[:3 * n].reshape(n, 3).copy()
            self._results['converged'] = bool(_mod.job_opt_converged)

    def copy(self):
        """Return a new PyQuick with the same setup state.

        Results from a previous :meth:`run` are not copied — the new instance
        starts unrun.  All setup attributes (_calc, _basis, _methods, _geom)
        are independent copies, so changes to one object do not affect the other.
        """
        new = PyQuick()
        new._calc    = self._calc
        new._basis   = self._basis
        new._methods = list(self._methods)   # list of immutable tuples — shallow copy is sufficient
        new._geom    = self._geom
        return new

    def __copy__(self):
        return self.copy()

    def __del__(self):
        # Only finalize QUICK if this object successfully ran a calculation.
        try:
            if self._ran and _mod.job_active:
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
        """QM <-> external point-charge interaction energy in Hartree.

        Only available when EXTCHARGES was set. This term is reported
        separately and is never part of ``total_energy``.
        """
        self._require_run('e_external_charge')
        if 'e_external_charge' not in self._results:
            raise AttributeError(
                "'e_external_charge' is unavailable: EXTCHARGES was not set "
                "for this job — enable it via set_method('EXTCHARGES')"
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
        """Alpha density matrix as a numpy array of shape (nbasis, nbasis)."""
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
# High-level Calculation / Result API
#
# Thin wrappers over the low-level PyQuick engine above.  The
# engine keeps the canonical pyquick conventions (check()-style errors, in-memory
# transfer, per-run snapshotting); Calculation/Result are the documented surface.
# ---------------------------------------------------------------------------

# Log/output file stem used when a run is given no name. Mirrors the default of
# `output_stem` in pyquick.f90.
_DEFAULT_LOG_STEM = 'pyquick_job'


def _count_atoms(geometry):
    """Number of atom lines ('SYMBOL X Y Z') in a geometry string.

    A light count for the geo_opt atom-count guard, not a full parse — QUICK
    still validates the geometry itself in read_geom.
    """
    return sum(1 for line in str(geometry).splitlines() if len(line.split()) >= 4)

# The energy breakdown is ALWAYS returned (every job computes it), so it is not
# a "property" the caller opts into.
_ENERGY_ATTRS = ('total_energy', 'nuclear_repulsion', 'e_electronic',
                 'e_one_electron', 'e_two_electron', 'e_xc')

# Opt-in extras the caller selects via `properties`.  These cost extra work
# beyond the SCF (they need the DIPOLE keyword), so they are not returned unless
# asked for.
_SUPPORTED_PROPERTIES = {
    'mulliken_charges',  # needs the DIPOLE keyword to be computed
    'lowdin_charges',    # needs the DIPOLE keyword to be computed
    'dipole',            # dipole moment vector (Debye)
}

# Always returned on the Result because every SCF computes them for free
# (no extra keyword, no extra routine).  Accepted in `properties` for backward
# compatibility, but they are a no-op there — you get them either way.
_ALWAYS_ON_PROPERTIES = {
    'mo_energies',       # alpha MO energies
    'density_matrix',    # alpha density matrix
}

# These are all computed inside QUICK's `dipole` routine, which only runs when
# the DIPOLE keyword is set.
_NEEDS_DIPOLE = {'mulliken_charges', 'lowdin_charges', 'dipole'}

# Job-type words: not properties.  The job type is the method you call.
_JOBTYPE_WORDS = {'energy', 'gradient', 'gradients', 'forces',
                  'optimize', 'optimized_coords', 'optimized_coordinates', 'geo_opt'}

# Extras planned for later phases — reported with a clearer message than
# "unknown property" so callers know they exist but aren't wired up yet.
_DEFERRED_PROPERTIES = {
    'beta_mo_energies': 'Phase 3',
    'beta_density_matrix': 'Phase 3',
    'mp2': 'Phase 3',
    'e_mp2': 'Phase 3',
    'esp_charges': 'Phase 4',
}

# opt-in property name -> Result attributes it unlocks.  The conditional energies
# (e_dispersion, e_external_charge) are gated by input keywords, not by the
# properties list — see Result.
_PROPERTY_ATTRS = {
    'mulliken_charges': ('mulliken',),
    'lowdin_charges': ('lowdin',),
    'dipole': ('dipole',),
}


def _resolve_method(method, mult):
    """Map a user method + multiplicity to (calc_keyword, functional_token).

    'HF'/'RHF'/'UHF' are explicit Hartree-Fock; any other name is treated as a
    DFT functional and routed through DFT/UDFT.  mult > 1 selects the
    unrestricted path.
    """
    if not method or not str(method).strip():
        raise ValueError("Calculation requires a 'method' (e.g. 'HF' or 'B3LYP')")
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


class Calculation:
    """A reusable set of QUICK calculation settings.

    The job type is chosen by which method you call:

        calc.get_energy(geometry)  -> single-point energy
        calc.get_grad(geometry)    -> energy + forces           (Phase 2)
        calc.geo_opt(geometry)     -> optimized geometry         (Phase 2)

    Example::

        calc = Calculation(method='B3LYP', basis='6-31G*',
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
        Calculation append; an existing file is backed up to ``<file>~``.
        ``log=False`` sends the log to the platform null device, so the run leaves
        nothing on disk.
    """

    def __init__(self, method, basis, properties=(),
                 charge=0, mult=1, keywords=None, log=True):
        if not basis or not str(basis).strip():
            raise ValueError("Calculation requires a 'basis' (e.g. '6-31G*')")

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
            if p in _DEFERRED_PROPERTIES:
                raise ValueError(
                    f"property {p!r} is not available in this version "
                    f"(planned for {_DEFERRED_PROPERTIES[p]}); "
                    f"supported: {sorted(_SUPPORTED_PROPERTIES)}")
            raise ValueError(
                f"unknown property {p!r}; supported: {sorted(_SUPPORTED_PROPERTIES)}")

        self.method = method
        self.basis = basis
        self.properties = requested
        self.charge = int(charge)
        self.mult = int(mult)
        self.keywords = dict(keywords) if keywords else {}
        self.log = bool(log)
        self._calc_keyword, self._functional = _resolve_method(method, self.mult)
        self._engine = None

    def _build_engine(self):
        job = PyQuick()
        job.set_calc(self._calc_keyword)
        job.set_basis(self.basis)
        if self._functional is not None:
            job.set_method(self._functional)
        if any(p in _NEEDS_DIPOLE for p in self.properties):
            job.set_method('DIPOLE')
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

    def _prepare(self, geometry, name):
        """Shared setup for a run: engine, output stem, geometry."""
        if self._engine is None:
            self._engine = self._build_engine()
        job = self._engine
        # Set the stem on every run, not just when a name is given: it is Fortran
        # module state that would otherwise persist from an earlier run (or from
        # another Calculation), silently sending this run's log to that file.
        job.set_output(str(name) if name is not None else _DEFAULT_LOG_STEM)
        job.read_geom(geometry)
        return job

    def get_energy(self, geometry, name=None):
        """Single-point energy on *geometry*; returns a :class:`Result`.

        *geometry* is a 'SYMBOL X Y Z' multi-line string (Angstrom).  *name*
        labels the run (it lands in ``Result.metadata['name']``) and, when
        logging is on, names the log file — ``name='water'`` -> ``water.out``.
        A single engine is reused across successive calls on the same
        Calculation, so each call fully finalizes the previous one inside QUICK.
        """
        job = self._prepare(geometry, name)
        job.run(jobtype=0, log=self.log)
        return Result._from_engine(job, self, geometry, name)

    def get_grad(self, geometry, name=None):
        """Energy + nuclear gradient on *geometry*; returns a :class:`Result`.

        The gradient job runs the SCF and the gradient together, so the returned
        Result carries **both** the energy breakdown and ``gradient`` (dE/dR,
        shape ``(natom, 3)``, Hartree/Bohr) — the energy is not computed twice.
        *name* labels the run and, when logging is on, names the log file.
        """
        job = self._prepare(geometry, name)
        job.run(jobtype=1, log=self.log)
        return Result._from_engine(job, self, geometry, name)

    def geo_opt(self, geometry, name=None, max_cycles=None):
        """Optimize *geometry*; returns a :class:`Result`.

        Starting from *geometry*, QUICK relaxes the structure and the Result
        carries ``optimized_coordinates`` (final structure, Angstrom),
        ``converged`` (did it reach the convergence criteria, or just run out of
        cycles?), and the energy breakdown **of the optimized structure**.
        ``coordinates`` still holds the geometry you passed in.

        *max_cycles* caps the optimization; the default (``None``) runs to
        convergence with no cap, which is QUICK's own behaviour.

        The optimizer is QUICK's default (DL-Find) unless you add the Cartesian
        optimizer via ``keywords={'LOPT': None}``.  DL-Find does not support
        molecules with fewer than 3 atoms — for those, use ``LOPT`` (this raises
        a ``ValueError`` otherwise).

        Always check ``result.converged`` — a run that hits *max_cycles* returns
        the last step's geometry, which is **not** optimized.
        """
        # DL-Find (the default) fails on < 3 atoms; steer the user to LOPT.
        using_lopt = any(str(k).upper() == 'LOPT' for k in self.keywords)
        if not using_lopt and _count_atoms(geometry) < 3:
            raise ValueError(
                "DL-Find (the default geometry optimizer) does not support "
                "molecules with fewer than 3 atoms. Use the Cartesian optimizer: "
                "Calculation(..., keywords={'LOPT': None}).")

        job = self._prepare(geometry, name)
        job.run(jobtype=2, log=self.log,
                max_cycles=0 if max_cycles is None else int(max_cycles))
        return Result._from_engine(job, self, geometry, name)

    @property
    def input_string(self):
        """The assembled QUICK keyword line + basis (no geometry until run)."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine.input_string


class Result:
    """Results of a completed :class:`Calculation` run.

    Properties are accessed as attributes.  Accessing one that was not requested
    in ``Calculation.properties`` (or could not be computed) raises
    ``AttributeError``.
    """

    def __init__(self, values, requested, metadata):
        self._values = dict(values)
        self._requested = set(requested)
        self.metadata = dict(metadata)

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
        for key in ('mulliken', 'lowdin', 'mo_energies', 'density_matrix',
                    'dipole', 'gradient', 'optimized_coordinates', 'converged'):
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
                f"calculation — add '{prop}' to Calculation.properties")
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
        """One-electron energy (kinetic + nuclear-electron attraction) in Hartree."""
        return self._values['e_one_electron']

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
                "e.g. Calculation(..., keywords={'D3': None})")
        return self._values['e_dispersion']

    @property
    def e_external_charge(self):
        """QM <-> external point-charge interaction energy in Hartree.

        Only available when EXTCHARGES was set in the input.  Reported
        separately; never included in ``total_energy``.
        """
        if 'e_external_charge' not in self._values:
            raise AttributeError(
                "'e_external_charge' is unavailable: EXTCHARGES was not set "
                "for this calculation — add it via "
                "Calculation(..., keywords={'EXTCHARGES': None})")
        return self._values['e_external_charge']

    # -- always available (every SCF computes them) --------------------------
    @property
    def mo_energies(self):
        """Alpha molecular orbital energies, numpy array of shape (NBSuse,)."""
        if 'mo_energies' not in self._values:
            raise AttributeError("'mo_energies' was not computed for this run")
        return self._values['mo_energies']

    @property
    def density_matrix(self):
        """Alpha density matrix, numpy array of shape (nbasis, nbasis)."""
        if 'density_matrix' not in self._values:
            raise AttributeError("'density_matrix' was not computed for this run")
        return self._values['density_matrix']

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

    # -- gradient (produced by get_grad, not by a `properties` request) -------
    @property
    def gradient(self):
        """Nuclear gradient dE/dR, numpy array of shape (natom, 3), Hartree/Bohr.

        Only present on a Result from :meth:`Calculation.get_grad`.
        """
        if 'gradient' not in self._values:
            raise AttributeError(
                "'gradient' is unavailable: this Result came from get_energy() — "
                "use calc.get_grad(geometry) to compute the nuclear gradient")
        return self._values['gradient']

    # -- optimization (produced by geo_opt) ----------------------------------
    @property
    def optimized_coordinates(self):
        """Optimized atomic coordinates in Angstrom, shape (natom, 3).

        Only present on a Result from :meth:`Calculation.geo_opt`.  Check
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
