module pyquick

#include "util.fh"
    use iso_fortran_env, only : output_unit
    use quick_molspec_module, only : quick_molspec, natom, xyz, alloc
    use quick_calculated_module, only : quick_qm_struct
    use quick_method_module, only : quick_method
    use quick_files_module, only : iOutFile, outFileName, inFileName, isTemplate, &
                                   set_quick_files, print_quick_io_file
    use quick_api_module, only : quick_api
    use quick_constants_module, only : SYMBOL, SYMBOL_MAX, A_TO_BOHRS
    use quick_basis_module, only : nbasis, NBSuse
    use quick_cutoff_module, only : schwarzoff
    use quick_eri_cshell_module, only : getEriPrecomputables
    use quick_sad_guess_module, only : getSadGuess
    use quick_grad_cshell_module, only : cshell_gradient
    use quick_grad_oshell_module, only : oshell_gradient
    use quick_optimizer_module, only : lopt
    use quick_exception_module, only : get_exception_message

    implicit none

    private
    public :: set_calc, set_basis, set_method, clear_methods, read_geom, &
              had_error, error_message, print_input, input_string, &
              job_run, job_destroy, job_set_output, &
              job_open, job_step, job_close, &
              job_active, &
              job_total_energy, job_e_core, job_e_electronic, &
              job_e_1e, job_e_xc, job_e_disp, job_e_charge, &
              job_dipole, job_has_dipole, &
              job_has_dispersion, job_has_extcharge, job_has_gradient, &
              job_has_mulliken, job_has_lowdin, job_has_mo_energies, job_has_density_matrix, &
              job_get_mulliken, job_get_lowdin, job_get_mo_energies, job_get_density_matrix, &
              job_get_gradients, job_get_geometry, job_exception_message, &
              job_has_optimized, job_opt_converged, job_get_optimized_geometry

    integer, parameter :: KEYWORD_LEN = 300
    integer, parameter :: INPUT_LEN   = 10000

    character(len=8) :: calc_keyword = ''
    character(len=:), allocatable :: basis_token
    integer, allocatable          :: geom_atnum(:)      ! atomic numbers, length geom_natom
    double precision, allocatable :: geom_coords(:,:)   ! Angstrom coordinates, shape (3, geom_natom)

    logical :: has_calc  = .false.
    logical :: has_basis = .false.
    logical :: has_geom  = .false.

    logical            :: had_error     = .false.
    character(len=512) :: error_message = ''

    character(len=INPUT_LEN) :: input_string = ''

    ! atom count stored by read_geom so job_run knows natom before alloc
    integer :: geom_natom = 0

    ! output file stem (default 'pyquick_job')
    character(len=80) :: output_stem = 'pyquick_job'

    ! job state
    logical :: job_active = .false.

    ! a persistent job (job_open .. repeated job_step .. job_close) is set up:
    ! the molecule and basis are allocated once and the density is reused across
    ! steps (only coordinates change), until job_close frees everything.
    logical :: job_open_flag = .false.

    ! scalar results (always available after a successful job_run)
    ! job_e_core is QUICK's ECore: the core-core NUCLEAR REPULSION energy
    double precision :: job_total_energy   = 0.0d0
    double precision :: job_e_core         = 0.0d0
    double precision :: job_e_electronic   = 0.0d0
    double precision :: job_e_1e           = 0.0d0
    double precision :: job_e_xc           = 0.0d0
    double precision :: job_e_disp         = 0.0d0
    double precision :: job_e_charge       = 0.0d0   ! ECharge; never part of ETot

    ! dipole moment vector (Debye), available when the DIPOLE routine ran
    double precision :: job_dipole(3) = 0.0d0
    logical :: job_has_dipole = .false.

    ! whether dispersion (D* keyword) / EXTCHARGES were active for the last run
    logical :: job_has_dispersion = .false.
    logical :: job_has_extcharge  = .false.

    ! whether a nuclear gradient was computed for the last run
    logical :: job_has_gradient = .false.

    ! geometry optimization: whether the last run was an OPT job, and whether it
    ! reached convergence (as opposed to stopping at the cycle limit)
    logical :: job_has_optimized  = .false.
    logical :: job_opt_converged  = .false.

    ! availability flags for array results
    logical :: job_has_mulliken      = .false.
    logical :: job_has_lowdin        = .false.
    logical :: job_has_mo_energies   = .false.
    logical :: job_has_density_matrix = .false.

    type :: method_entry
        character(len=:), allocatable :: keyword
        character(len=:), allocatable :: arg
    end type method_entry

    type(method_entry), allocatable :: method_list(:)

contains

    ! -----------------------------------------------------------------------
    ! Backward-compatible input-assembly API
    ! -----------------------------------------------------------------------

    subroutine set_calc(keyword)
        character(len=*), intent(in) :: keyword
        character(len=:), allocatable :: normalized

        normalized = uppercase(trim(keyword))

        select case (trim(normalized))
        case ('HF', 'UHF', 'DFT', 'UDFT')
            calc_keyword = normalized
            has_calc = .true.
            call rebuild_input()
        case default
            call fail('set_calc: keyword must be HF, UHF, DFT or UDFT')
        end select
    end subroutine set_calc

    subroutine set_basis(basis_name)
        character(len=*), intent(in) :: basis_name

        if (len_trim(basis_name) == 0) then
            call fail('set_basis: basis name must be non-empty')
            return
        end if

        basis_token = 'BASIS=' // trim(adjustl(basis_name))
        has_basis = .true.
        call rebuild_input()
    end subroutine set_basis

    subroutine set_method(keyword, arg)
        character(len=*), intent(in) :: keyword
        character(len=*), intent(in), optional :: arg
        character(len=:), allocatable :: uname
        integer :: i

        if (len_trim(keyword) == 0) then
            call fail('set_method: keyword name must be non-empty')
            return
        end if

        uname = uppercase(trim(adjustl(keyword)))

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (trim(method_list(i)%keyword) == trim(uname)) then
                    if (present(arg) .and. len_trim(arg) > 0) then
                        method_list(i)%arg = trim(adjustl(arg))
                    else
                        method_list(i)%arg = ''
                    end if
                    call rebuild_input()
                    return
                end if
            end do
        end if

        call append_method(uname, arg)
        call rebuild_input()
    end subroutine set_method

    subroutine read_geom(input)
        character(len=*), intent(in) :: input
        character(len=:), allocatable :: line
        integer :: start, len_input, nl, atom_count, ios, k, z
        character(len=4) :: sym
        double precision :: cx, cy, cz

        ! discard any previous geometry so a second call replaces the first
        if (allocated(geom_atnum))  deallocate(geom_atnum)
        if (allocated(geom_coords)) deallocate(geom_coords)
        geom_natom = 0
        has_geom   = .false.

        ! --- first pass: count non-empty lines to know how many atoms ---
        atom_count = 0
        start      = 1
        len_input  = len(input)

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                if (len_trim(input(start:)) > 0) atom_count = atom_count + 1
                exit
            else
                if (len_trim(input(start:start+nl-2)) > 0) atom_count = atom_count + 1
                start = start + nl
            end if
        end do

        if (atom_count == 0) then
            call fail('read_geom: geometry must contain at least one atom')
            return
        end if

        allocate(geom_atnum(atom_count))
        allocate(geom_coords(3, atom_count))

        ! --- second pass: parse and validate each line ---
        atom_count = 0
        start      = 1

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                line = trim(adjustl(input(start:)))
            else
                line = trim(adjustl(input(start:start+nl-2)))
                start = start + nl
            end if

            if (len_trim(line) == 0) then
                if (nl == 0) exit
                cycle
            end if

            ! parse symbol and three coordinates; accepts decimal and scientific notation
            sym = ''
            read(line, *, iostat=ios) sym, cx, cy, cz
            if (ios /= 0) then
                call fail('read_geom: cannot parse line (expected: SYMBOL X Y Z): ' // &
                          trim(line))
                return
            end if

            ! validate symbol against the QUICK element table
            z = 0
            do k = 1, SYMBOL_MAX
                if (trim(uppercase(sym)) == trim(uppercase(SYMBOL(k)))) then
                    z = k
                    exit
                end if
            end do
            if (z == 0) then
                call fail('read_geom: unknown element symbol "' // trim(sym) // &
                          '" in line: ' // trim(line))
                return
            end if

            atom_count = atom_count + 1
            geom_atnum(atom_count)     = z
            geom_coords(1, atom_count) = cx
            geom_coords(2, atom_count) = cy
            geom_coords(3, atom_count) = cz

            if (nl == 0) exit
        end do

        geom_natom = atom_count
        has_geom   = .true.
        call rebuild_input()

    end subroutine read_geom

    subroutine print_input()
        write(output_unit, '(A)') trim(input_string)
    end subroutine print_input

    subroutine rebuild_input()
        character(len=:), allocatable :: text
        character(len=64) :: atom_line
        integer :: i

        if (has_calc) then
            text = trim(calc_keyword)
        else
            text = ''
        end if

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(text) > 0) then
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                               // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(text) // ' ' // trim(method_list(i)%keyword)
                    end if
                else
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(method_list(i)%keyword) // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(method_list(i)%keyword)
                    end if
                end if
            end do
        end if

        if (has_basis) then
            if (len_trim(text) > 0) then
                text = trim(text) // ' ' // trim(basis_token)
            else
                text = trim(basis_token)
            end if
        end if

        if (has_geom) then
            do i = 1, geom_natom
                write(atom_line, '(A2, 3(1X, F12.6))') &
                    trim(SYMBOL(geom_atnum(i))), &
                    geom_coords(1,i), geom_coords(2,i), geom_coords(3,i)
                text = trim(text) // new_line('a') // trim(atom_line)
            end do
        end if

        input_string = ''
        if (len_trim(text) > 0) then
            input_string(1:min(len_trim(text), INPUT_LEN)) = &
                text(1:min(len_trim(text), INPUT_LEN))
        end if
    end subroutine rebuild_input

    ! -----------------------------------------------------------------------
    ! Job execution API
    ! -----------------------------------------------------------------------

    subroutine job_set_output(stem)
        character(len=*), intent(in) :: stem
        if (len_trim(stem) == 0) then
            call fail('job_set_output: stem must be non-empty')
            return
        end if
        output_stem = trim(stem)
    end subroutine job_set_output

    subroutine job_run(jobtype, do_log, max_cycles)
        ! jobtype: 0 = single-point energy
        !          1 = energy + nuclear gradient
        !          2 = geometry optimization
        ! do_log:  1 = write QUICK's log to <output_stem>.out (default behaviour),
        !          0 = send it to the platform null device, leaving no file.
        ! max_cycles: for jobtype 2 only -- <= 0 emits a bare OPTIMIZE keyword
        !          (QUICK's default: run to convergence, no cap); > 0 emits
        !          OPTIMIZE=<n> to cap the number of optimization cycles.
#if defined(GPU)
        ! GPU builds only: allmod exposes the gpu_* device routines and the basis
        ! arrays the uploads need. Compiled out on the CPU build (needs the build
        ! to preprocess this file: -cpp, and -DGPU for GPU builds).
        use allmod
#endif
        integer, intent(in) :: jobtype
        integer, intent(in) :: do_log
        integer, intent(in) :: max_cycles
        ! External free subroutines in libquick.so
        external :: initialize1, read_Job_and_Atom, getMol, getEnergy, dipole
        external :: finalize, outputCopyright, PrtDate, quick_open, dl_find

        character(len=:), allocatable :: keyword_line
        character(len=:), allocatable :: null_dev
        character(len=10) :: kwlen_str
        character(len=12) :: cyc_str
        character(len=1) :: open_mode
        integer :: ierr, i, j, k, ios
        integer :: natm_type
        integer :: atm_type_id(geom_natom)
        logical :: new_type, unit_open
        character(len=256) :: note

        ierr = 0

        ! --- decide the log open mode before finalize clears job_active ---
        ! first run replaces the file; re-runs append to it
        if (job_active) then
            open_mode = 'A'
        else
            open_mode = 'R'
        end if

        ! --- if a previous run is still active, finalize it before re-running ---
        ! This deallocates all basis/MO/density arrays sized for the previous run
        ! so they can be reallocated at the correct dimensions for the new run.
        if (job_active) then
            call finalize(iOutFile, ierr, 1)
            if (ierr /= 0) then
                call fail('job_run: finalize of previous run failed')
                return
            end if
            job_active = .false.
        end if

        ! --- validate prerequisites ---
        if (.not. has_calc) then
            call fail('job_run: call set_calc before run')
            return
        end if
        if (.not. has_basis) then
            call fail('job_run: call set_basis before run')
            return
        end if
        if (.not. has_geom) then
            call fail('job_run: call read_geom before run')
            return
        end if

        ! --- build keyword line ---
        keyword_line = build_keyword_line()

        ! for a gradient job, add the GRADIENT keyword so read_Job_and_Atom sets
        ! quick_method%grad before getMol allocates quick_qm_struct%gradient
        if (jobtype == 1) keyword_line = trim(keyword_line) // ' GRADIENT'

        ! for an optimization, OPTIMIZE sets quick_method%opt (and %grad). A bare
        ! keyword leaves iopt=0, which QUICK treats as "no cap, run to
        ! convergence"; OPTIMIZE=<n> caps the cycles. The optimizer is QUICK's
        ! default (DL-Find) unless the user added the LOPT keyword (Cartesian).
        if (jobtype == 2) then
            if (max_cycles > 0) then
                write(cyc_str, '(I0)') max_cycles
                keyword_line = trim(keyword_line) // ' OPTIMIZE=' // trim(cyc_str)
            else
                keyword_line = trim(keyword_line) // ' OPTIMIZE'
            end if
        end if

        ! --- guard against silent Fortran truncation on assignment to quick_api%Keywd ---
        if (len_trim(keyword_line) > KEYWORD_LEN) then
            write(kwlen_str, '(I0)') KEYWORD_LEN
            call fail('job_run: keyword line exceeds maximum length of ' // &
                      trim(kwlen_str) // ' characters')
            return
        end if

        ! --- configure quick_api for keyword injection ---
        quick_api%apiMode  = .true.
        quick_api%hasKeywd = .true.
        quick_api%Keywd    = trim(keyword_line)

        ! --- configure file names; isTemplate suppresses coord read in getMol ---
        inFileName = trim(output_stem) // '.in'
        isTemplate = .true.

        ! --- QUICK call chain (follows main.f90) ---
        call initialize1(ierr)
        if (ierr /= 0) then
            call fail('job_run: initialize1 failed')
            return
        end if

        call set_quick_files(.true., ierr)
        if (ierr /= 0) then
            call fail('job_run: set_quick_files failed')
            return
        end if

        ! --- open QUICK's diagnostic log ---
        if (do_log /= 0) then
            ! Normal case: a real <output_stem>.out file, as set_quick_files
            ! derived it. quick_open backs up any pre-existing file to '<file>~'.
            call quick_open(iOutFile, outFileName, 'U', 'F', open_mode, .false., ierr)
            if (ierr /= 0) then
                call fail('job_run: quick_open failed')
                return
            end if
        else
            ! Opted out: send the log to the platform null device so the run
            ! leaves nothing on disk. Open the unit directly rather than through
            ! quick_open, which would try to back up the device with 'mv'. The
            ! device name is chosen at runtime so this also works on Windows (NUL).
            null_dev = null_device()
            outFileName = null_dev
            inquire(unit=iOutFile, opened=unit_open)
            if (unit_open) close(iOutFile)
            open(unit=iOutFile, file=null_dev, status='UNKNOWN', form='FORMATTED', &
                 action='WRITE', iostat=ios)
            if (ios /= 0) then
                call fail('job_run: could not open null output device')
                return
            end if
        end if

        call outputCopyright(iOutFile, ierr)
        if (ierr /= 0) then
            call fail('job_run: outputCopyright failed')
            return
        end if

        note = 'TASK STARTS ON:'
        call PrtDate(iOutFile, note, ierr)
        if (ierr /= 0) then
            call fail('job_run: PrtDate failed')
            return
        end if

        call print_quick_io_file(iOutFile, ierr)
        if (ierr /= 0) then
            call fail('job_run: print_quick_io_file failed')
            return
        end if

#if defined(GPU)
        ! --- GPU: create context and pick a device (mirrors main.f90) ---
        call gpu_new(ierr)
        call gpu_init_device(ierr)
        call gpu_write_info(iOutFile, ierr)
#endif

        ! reads keyword from quick_api%Keywd; skips coordinate read (apiMode)
        call read_Job_and_Atom(ierr)
        if (ierr /= 0) then
            call fail('job_run: read_Job_and_Atom failed')
            return
        end if

        ! set natom (module-level target) BEFORE alloc uses it
        natom = geom_natom

        call alloc(quick_molspec, .false., ierr)
        if (ierr /= 0) then
            call fail('job_run: alloc(quick_molspec) failed')
            return
        end if

        ! --- inject geometry into QUICK module-level state ---

        ! build atom type list (deduplicate by atomic number)
        natm_type = 0
        atm_type_id = 0
        do i = 1, geom_natom
            new_type = .true.
            do k = 1, natm_type
                if (atm_type_id(k) == geom_atnum(i)) then
                    new_type = .false.
                    exit
                end if
            end do
            if (new_type) then
                natm_type = natm_type + 1
                atm_type_id(natm_type) = geom_atnum(i)
            end if
        end do

        quick_molspec%iAtomType = natm_type
        do i = 1, natm_type
            quick_molspec%atom_type_sym(i) = SYMBOL(atm_type_id(i))
        end do

        ! inject atomic numbers and coordinates (convert Angstrom -> Bohr)
        do i = 1, geom_natom
            quick_molspec%iattype(i) = geom_atnum(i)
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do
        quick_molspec%xyz => xyz

        ! --- initial guess ---
        if (quick_method%SAD) then
            call getSadGuess(ierr)
            if (ierr /= 0) then
                call fail('job_run: getSadGuess failed')
                return
            end if
        end if

        ! --- build molecular orbital / basis information ---
        call getMol(ierr)
        if (ierr /= 0) then
            call fail('job_run: getMol failed')
            return
        end if

#if defined(GPU)
        ! --- GPU: allocate scratch, upload method + molecule/coords ---
        call gpu_allocate_scratch(quick_method%grad .or. quick_method%opt)
        call upload(quick_method, ierr)
        if (.not. quick_method%opt) then
            call gpu_setup(natom, nbasis, quick_molspec%nElec, quick_molspec%imult, &
                           quick_molspec%molchg, quick_molspec%iAtomType)
            call gpu_upload_xyz(xyz)
            call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
        end if
#endif

        ! --- ERI precomputables and cutoff screening ---
        call getEriPrecomputables()
        call schwarzoff()

#if defined(GPU)
        ! --- GPU: upload basis + Schwarz cutoffs (needs the arrays just built) ---
        if (.not. quick_method%opt) then
            call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
                ncontract, itype, aexp, dcoeff, &
                quick_basis%first_basis_function, quick_basis%last_basis_function, &
                quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
                quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
                quick_basis%ktype, quick_basis%kprim, quick_basis%kshell, quick_basis%Ksumtype, &
                quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, &
                quick_basis%Qsbasis, quick_basis%Qfbasis, &
                quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
            call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)
            call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, &
                                quick_molspec%extchg, ierr)
        end if
#endif

        ! --- energy, gradient, or optimization ---
        ! quick_method%opt/%grad come from the OPTIMIZE/GRADIENT keywords appended
        ! above. Each branch runs its own SCF, so getEnergy is only called for a
        ! plain energy job (this mirrors quick_api_module / main.f90).
        if (quick_method%opt) then
            ! DL-Find is QUICK's default optimizer; the LOPT keyword selects the
            ! Cartesian optimizer instead. DL-Find crashes on molecules with fewer
            ! than 3 atoms, so refuse that combination with a clear message rather
            ! than segfaulting (the Python layer guards this too).
            if (quick_method%usedlfind) then
                if (geom_natom < 3) then
                    call fail('geo_opt: DL-Find does not support molecules with ' // &
                              'fewer than 3 atoms; use the Cartesian optimizer ' // &
                              '(add the LOPT keyword)')
                    return
                end if
                call dl_find(ierr, .true.)
            else
                call lopt(ierr)
            end if
            if (ierr /= 0) then
                call fail('job_run: geometry optimization failed')
                return
            end if
        else if (quick_method%grad) then
            if (quick_method%unrst) then
                call oshell_gradient(ierr)
            else
                call cshell_gradient(ierr)
            end if
            if (ierr /= 0) then
                call fail('job_run: gradient calculation failed')
                return
            end if
        else
            call getEnergy(.false., ierr)
            if (ierr /= 0) then
                call fail('job_run: getEnergy failed')
                return
            end if
        end if

        ! --- post-SCF charges/dipole + harvest all results ---
        call harvest_results(quick_method%grad)

#if defined(GPU)
        ! --- GPU: free device scratch and context ---
        call gpu_deallocate_scratch(quick_method%grad .or. quick_method%opt)
        call gpu_delete(ierr)
#endif

        job_active = .true.

    end subroutine job_run

    ! -----------------------------------------------------------------------
    ! Persistent job: set up once (job_open), run many geometries of the same
    ! molecule (job_step, reusing the density), then free everything (job_close).
    ! -----------------------------------------------------------------------

    subroutine job_open(do_log)
        ! One-time setup for a fixed molecule. Prerequisites (as for job_run):
        ! set_calc, set_basis and read_geom (the initial geometry) must be done.
        ! do_log: 1 = write <stem>.out, 0 = null device.
#if defined(GPU)
        ! GPU builds only: allmod exposes the gpu_* device routines (use is
        ! subroutine-scoped in Fortran, so job_run's own `use allmod` doesn't
        ! cover this subroutine -- it needs its own).
        use allmod
#endif
        integer, intent(in) :: do_log
        external :: initialize1, read_Job_and_Atom, getMol
        external :: outputCopyright, PrtDate, quick_open

        character(len=:), allocatable :: keyword_line, null_dev
        character(len=10) :: kwlen_str
        integer :: ierr, i, j, k, ios, natm_type
        integer :: atm_type_id(geom_natom)
        logical :: new_type, unit_open
        character(len=256) :: note

        ierr = 0
        if (job_open_flag) then
            call fail('job_open: a job is already open; call job_close first')
            return
        end if
        if (.not. has_calc)  then; call fail('job_open: call set_calc before'); return; end if
        if (.not. has_basis) then; call fail('job_open: call set_basis before'); return; end if
        if (.not. has_geom)  then; call fail('job_open: call read_geom before'); return; end if

        ! Append GRADIENT so getMol sizes quick_qm_struct%gradient; individual
        ! steps then opt into a gradient without re-running getMol.
        keyword_line = trim(build_keyword_line()) // ' GRADIENT'
        if (len_trim(keyword_line) > KEYWORD_LEN) then
            write(kwlen_str, '(I0)') KEYWORD_LEN
            call fail('job_open: keyword line exceeds ' // trim(kwlen_str) // ' characters')
            return
        end if

        quick_api%apiMode  = .true.
        quick_api%hasKeywd = .true.
        quick_api%Keywd    = trim(keyword_line)
        inFileName = trim(output_stem) // '.in'
        isTemplate = .true.

        call initialize1(ierr)
        if (ierr /= 0) then; call fail('job_open: initialize1 failed'); return; end if
        call set_quick_files(.true., ierr)
        if (ierr /= 0) then; call fail('job_open: set_quick_files failed'); return; end if

        if (do_log /= 0) then
            call quick_open(iOutFile, outFileName, 'U', 'F', 'R', .false., ierr)
            if (ierr /= 0) then; call fail('job_open: quick_open failed'); return; end if
        else
            null_dev = null_device()
            outFileName = null_dev
            inquire(unit=iOutFile, opened=unit_open)
            if (unit_open) close(iOutFile)
            open(unit=iOutFile, file=null_dev, status='UNKNOWN', form='FORMATTED', &
                 action='WRITE', iostat=ios)
            if (ios /= 0) then; call fail('job_open: could not open null output device'); return; end if
        end if

        call outputCopyright(iOutFile, ierr)
        note = 'TASK STARTS ON:'
        call PrtDate(iOutFile, note, ierr)
        call print_quick_io_file(iOutFile, ierr)

#if defined(GPU)
        ! --- GPU: create context and pick a device (mirrors job_run) ---
        call gpu_new(ierr)
        call gpu_init_device(ierr)
        call gpu_write_info(iOutFile, ierr)
#endif

        call read_Job_and_Atom(ierr)
        if (ierr /= 0) then; call fail('job_open: read_Job_and_Atom failed'); return; end if

        natom = geom_natom
        call alloc(quick_molspec, .false., ierr)
        if (ierr /= 0) then; call fail('job_open: alloc(quick_molspec) failed'); return; end if

        ! atom types (deduplicate by atomic number) + initial coordinates
        natm_type = 0
        atm_type_id = 0
        do i = 1, geom_natom
            new_type = .true.
            do k = 1, natm_type
                if (atm_type_id(k) == geom_atnum(i)) then; new_type = .false.; exit; end if
            end do
            if (new_type) then; natm_type = natm_type + 1; atm_type_id(natm_type) = geom_atnum(i); end if
        end do
        quick_molspec%iAtomType = natm_type
        do i = 1, natm_type
            quick_molspec%atom_type_sym(i) = SYMBOL(atm_type_id(i))
        end do
        do i = 1, geom_natom
            quick_molspec%iattype(i) = geom_atnum(i)
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do
        quick_molspec%xyz => xyz

        if (quick_method%SAD) then
            call getSadGuess(ierr)
            if (ierr /= 0) then; call fail('job_open: getSadGuess failed'); return; end if
        end if
        call getMol(ierr)
        if (ierr /= 0) then; call fail('job_open: getMol failed'); return; end if

#if defined(GPU)
        ! --- GPU: allocate persistent scratch (always gradient-capable -- job_step
        ! toggles want_gradient per call, so this can't be sized in advance, unlike
        ! job_run's one-shot `quick_method%grad .or. quick_method%opt`) and upload
        ! the method settings once for the life of this job. ---
        call gpu_allocate_scratch(.true.)
        call upload(quick_method, ierr)
#endif

        job_open_flag = .true.
        job_active    = .true.
    end subroutine job_open

    subroutine job_step(want_gradient)
        ! Run one geometry of the open job. read_geom must have set the new
        ! coordinates (same atoms). want_gradient: 1 = also compute the gradient.
        ! getMol is NOT called, so quick_qm_struct%dense from the previous step
        ! survives and seeds this SCF.
#if defined(GPU)
        use allmod
#endif
        integer, intent(in) :: want_gradient
        integer :: ierr, i, j

        ierr = 0
        if (.not. job_open_flag) then
            call fail('job_step: no job is open; call job_open first')
            return
        end if
        if (geom_natom /= natom) then
            call fail('job_step: geometry has a different number of atoms than the open job')
            return
        end if

        ! update coordinates only (atoms are fixed for the job)
        do i = 1, natom
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do

        call getEriPrecomputables()
        call schwarzoff()

#if defined(GPU)
        ! --- GPU: re-upload geometry + basis for this step. gpu_upload_basis and
        ! gpu_upload_oei also bake in geometry-derived quantities (interatomic
        ! distances, Gaussian product centers), so this must repeat every step,
        ! not just once in job_open -- see quick_api_module's run_quick, which
        ! does the same full re-upload unconditionally on every call. ---
        call gpu_setup(natom, nbasis, quick_molspec%nElec, quick_molspec%imult, &
                       quick_molspec%molchg, quick_molspec%iAtomType)
        call gpu_upload_xyz(xyz)
        call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
        call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
            ncontract, itype, aexp, dcoeff, &
            quick_basis%first_basis_function, quick_basis%last_basis_function, &
            quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
            quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
            quick_basis%ktype, quick_basis%kprim, quick_basis%kshell, quick_basis%Ksumtype, &
            quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, &
            quick_basis%Qsbasis, quick_basis%Qfbasis, &
            quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
        call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)
        call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, &
                            quick_molspec%extchg, ierr)
#endif

        if (want_gradient /= 0) then
            if (quick_method%unrst) then
                call oshell_gradient(ierr)
            else
                call cshell_gradient(ierr)
            end if
            if (ierr /= 0) then; call fail('job_step: gradient calculation failed'); return; end if
        else
            call getEnergy(.false., ierr)
            if (ierr /= 0) then; call fail('job_step: getEnergy failed'); return; end if
        end if

#if defined(GPU)
        ! --- GPU: intentionally NOT calling gpu_cleanup() here. ---
        ! quick_api_module's run_quick calls gpu_cleanup() at this exact point to free
        ! the geometry-dependent basis/xyz arrays before the next step re-uploads them
        ! (see gpu_upload_molspecs). That call segfaults: confirmed reproducible even
        ! by adding the identical call to job_run's already-verified one-shot teardown
        ! (src/gpu/cuda/gpu.cu's gpu_cleanup_ crashes deep in its SAFE_DELETE sequence,
        ! in the gpu_basis field group, before ever reaching gpu_calculated/gpu_cutoff --
        ! a pre-existing bug in gpu_cleanup()/gpu_buffer_type, not something introduced
        ! here; job_run never exercises it since it tears down the whole context instead).
        ! Trade-off accepted for now: each job_step's gpu_upload_basis/gpu_setup calls
        ! re-`new` their device buffers without freeing the previous step's, so a
        ! persistent GPU job leaks a basis-sized chunk of device memory per step. Bounded
        ! by basis size (not scratch/SCF-iteration size) and freed at job_close/job_destroy
        ! via gpu_delete's cudaDeviceReset. Fine for short-to-moderate step counts; will
        ! exhaust device memory on very long-running jobs. Follow-up: root-cause and fix
        ! gpu_cleanup() itself, then call it here instead of skipping it.
#endif

        call harvest_results(want_gradient /= 0)
    end subroutine job_step

    subroutine job_close()
        external :: finalize
        integer :: ierr
        ierr = 0
        if (job_active) then
            ! Only tear down the GPU context if THIS call opened one (job_open_flag).
            ! job_active is also left .true. by the one-shot job_run, which already
            ! tore its own GPU context down (gpu_delete) before returning -- calling
            ! job_gpu_teardown() again there would double-free the (still-dangling,
            ! gpu_delete_ never nulls the global `gpu` pointer) device context.
            if (job_open_flag) call job_gpu_teardown()
            call finalize(iOutFile, ierr, 1)
        end if
        job_active    = .false.
        job_open_flag = .false.
    end subroutine job_close

    subroutine job_gpu_teardown()
        ! Shared GPU-context teardown for job_close and job_destroy. Either one
        ! can be the last call on an open persistent job -- job_destroy also
        ! backstops __init__.py's PyQuick.__del__ -- so both must free the
        ! device context, not just job_close. Mirrors quick_api_module's
        ! delete_quick_job, including the libxc GPU cleanup that job_run's
        ! one-shot teardown skips (harmless there for non-DFT jobs, but a
        ! persistent job can run DFT).
#if defined(GPU)
        use allmod
        integer :: ierr
        ierr = 0
        call delete(quick_method, ierr)
        call gpu_deallocate_scratch(.true.)
        call gpu_delete(ierr)
#endif
    end subroutine job_gpu_teardown

    subroutine harvest_results(did_gradient)
        ! Copy energies/properties out of QUICK's module state after a run.
        external :: dipole
        logical, intent(in) :: did_gradient

        if (quick_method%dipole) call dipole

        job_total_energy   = quick_qm_struct%ETot
        job_e_core         = quick_qm_struct%ECore
        job_e_electronic   = quick_qm_struct%EEl
        job_e_1e           = quick_qm_struct%E1e
        job_e_xc           = quick_qm_struct%Exc
        job_e_disp         = quick_qm_struct%Edisp
        job_e_charge       = quick_qm_struct%ECharge
        job_has_dispersion = quick_method%edisp
        job_has_extcharge  = quick_method%extcharges

        job_has_dipole = quick_method%dipole
        if (quick_method%dipole) job_dipole = quick_qm_struct%dipole

        job_has_mulliken       = quick_method%dipole
        job_has_lowdin         = quick_method%dipole
        job_has_mo_energies    = allocated(quick_qm_struct%E)
        job_has_density_matrix = allocated(quick_qm_struct%dense)
        job_has_gradient       = did_gradient .and. allocated(quick_qm_struct%gradient)

        job_has_optimized = quick_method%opt
        job_opt_converged = quick_qm_struct%opt_converged
    end subroutine harvest_results

    subroutine job_destroy()
        external :: finalize
        integer :: ierr
        ierr = 0
        if (job_active) then
            ! See job_close: only tear down the GPU context if a persistent job
            ! (job_open) actually left one open.
            if (job_open_flag) call job_gpu_teardown()
            call finalize(iOutFile, ierr, 1)
            job_active    = .false.
            job_open_flag = .false.
        end if
    end subroutine job_destroy

    ! -----------------------------------------------------------------------
    ! Array result getters
    ! Each checks the availability flag and sets had_error if not computed.
    ! -----------------------------------------------------------------------

    subroutine job_get_mulliken(charges, n)
        ! f2py cannot use external module variables as C array bounds, so we
        ! use a literal upper bound and return the actual count in n.
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_mulliken) then
            call fail("'mulliken' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Mulliken(1:natom)
    end subroutine job_get_mulliken

    subroutine job_get_lowdin(charges, n)
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_lowdin) then
            call fail("'lowdin' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Lowdin(1:natom)
    end subroutine job_get_lowdin

    subroutine job_get_mo_energies(energies, n)
        !f2py intent(out) energies, n
        integer, intent(out) :: n
        double precision, intent(out) :: energies(10000)
        energies = 0.0d0
        if (.not. job_has_mo_energies) then
            call fail("'mo_energies' were not computed; run() must complete successfully")
            n = 0
            return
        end if
        n = NBSuse
        energies(1:NBSuse) = quick_qm_struct%E(1:NBSuse)
    end subroutine job_get_mo_energies

    subroutine job_get_density_matrix(dm, nr, nc)
        ! Returns the alpha density matrix as a 1D (row-major) array of length
        ! nr*nc = nbasis*nbasis.  Reshape in Python: dm.reshape(nr, nc).
        ! We cap at 3000*3000 = 9_000_000 elements; large basis sets are rare.
        !f2py intent(out) dm, nr, nc
        integer, intent(out) :: nr, nc
        double precision, intent(out) :: dm(9000000)
        integer :: i, j, idx
        dm = 0.0d0
        if (.not. job_has_density_matrix) then
            call fail("'density_matrix' was not computed; run() must complete successfully")
            nr = 0
            nc = 0
            return
        end if
        nr = nbasis
        nc = nbasis
        do j = 1, nbasis
            do i = 1, nbasis
                idx = (i - 1) * nbasis + j
                dm(idx) = quick_qm_struct%dense(i, j)
            end do
        end do
    end subroutine job_get_density_matrix

    subroutine job_get_gradients(g, n)
        ! Returns the nuclear gradient as a flat, row-major (per-atom) array of
        ! length 3*n = 3*natom: (dx1,dy1,dz1, dx2,dy2,dz2, ...), in Hartree/Bohr.
        ! Reshape in Python: g[:3*n].reshape(n, 3).
        !f2py intent(out) g, n
        integer, intent(out) :: n
        double precision, intent(out) :: g(30000)
        integer :: i
        g = 0.0d0
        if (.not. job_has_gradient) then
            call fail("'gradient' was not computed; use get_grad() (not get_energy())")
            n = 0
            return
        end if
        n = natom
        do i = 1, 3 * natom
            g(i) = quick_qm_struct%gradient(i)
        end do
    end subroutine job_get_gradients

    subroutine job_get_optimized_geometry(coords, n)
        ! Returns the geometry after a successful OPT job: QUICK optimizes
        ! quick_molspec%xyz in place, in Bohr, so convert back to Angstrom here
        ! to match the units read_geom accepts and job_get_geometry returns.
        ! Flat, row-major per atom: (x1,y1,z1, x2,y2,z2, ...); reshape (n, 3).
        !f2py intent(out) coords, n
        integer, intent(out) :: n
        double precision, intent(out) :: coords(30000)
        integer :: i, j
        coords = 0.0d0
        if (.not. job_has_optimized) then
            call fail("'optimized_coordinates' were not computed; use geo_opt()")
            n = 0
            return
        end if
        n = natom
        do i = 1, natom
            do j = 1, 3
                coords((i - 1) * 3 + j) = xyz(j, i) / A_TO_BOHRS
            end do
        end do
    end subroutine job_get_optimized_geometry

    subroutine job_get_geometry(atnums, coords, n)
        ! Returns QUICK's own parsed geometry: atomic numbers and Angstrom
        ! coordinates as stored by read_geom (no re-parsing on the Python side).
        ! coords is flattened row-major per atom: (x1,y1,z1, x2,y2,z2, ...);
        ! reshape in Python: coords[:3*n].reshape(n, 3).
        !f2py intent(out) atnums, coords, n
        integer, intent(out) :: n
        integer, intent(out) :: atnums(10000)
        double precision, intent(out) :: coords(30000)
        integer :: i, j
        atnums = 0
        coords = 0.0d0
        if (.not. has_geom) then
            call fail('job_get_geometry: no geometry set; call read_geom first')
            n = 0
            return
        end if
        n = geom_natom
        do i = 1, geom_natom
            atnums(i) = geom_atnum(i)
            do j = 1, 3
                coords((i - 1) * 3 + j) = geom_coords(j, i)
            end do
        end do
    end subroutine job_get_geometry

    ! -----------------------------------------------------------------------
    ! Private helpers
    ! -----------------------------------------------------------------------

    function build_keyword_line() result(text)
        character(len=:), allocatable :: text
        integer :: i

        text = trim(calc_keyword)

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(method_list(i)%arg) > 0) then
                    text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                           // '=' // trim(method_list(i)%arg)
                else
                    text = trim(text) // ' ' // trim(method_list(i)%keyword)
                end if
            end do
        end if

        if (has_basis) text = trim(text) // ' ' // trim(basis_token)

    end function build_keyword_line

    subroutine clear_methods()
        if (allocated(method_list)) deallocate(method_list)
        call rebuild_input()
    end subroutine clear_methods

    subroutine append_method(uname, arg)
        character(len=*), intent(in) :: uname
        character(len=*), intent(in), optional :: arg
        type(method_entry), allocatable :: tmp(:)
        integer :: n

        if (.not. allocated(method_list)) then
            allocate(method_list(1))
            n = 1
        else
            n = size(method_list) + 1
            allocate(tmp(n))
            tmp(1:n-1) = method_list
            call move_alloc(tmp, method_list)
        end if

        method_list(n)%keyword = trim(uname)
        if (present(arg) .and. len_trim(arg) > 0) then
            method_list(n)%arg = trim(adjustl(arg))
        else
            method_list(n)%arg = ''
        end if
    end subroutine append_method

    subroutine fail(message)
        character(len=*), intent(in) :: message
        had_error     = .true.
        error_message = trim(message)
    end subroutine fail

    subroutine job_exception_message(code, msg)
        ! Expose QUICK's error text for a code, so Python can report/inspect it.
        !f2py intent(out) msg
        integer, intent(in) :: code
        character(len=200), intent(out) :: msg
        call get_exception_message(code, msg)
    end subroutine job_exception_message

    function uppercase(text) result(upper)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: upper
        integer :: i

        upper = text
        do i = 1, len(text)
            select case (upper(i:i))
            case ('a':'z')
                upper(i:i) = achar(iachar(upper(i:i)) - 32)
            end select
        end do
    end function uppercase

    function null_device() result(dev)
        ! Platform-appropriate null device for discarding QUICK's log.
        ! Detected at runtime so the choice does not depend on preprocessor
        ! flags or a specific Fortran compiler's predefined macros:
        !   Windows sets OS=Windows_NT and defines %WINDIR% -> use 'NUL'
        !   Unix/macOS -> use '/dev/null'
        character(len=:), allocatable :: dev
        character(len=64) :: val
        integer :: length, stat

        call get_environment_variable('OS', val, length, stat)
        if (stat == 0 .and. length > 0) then
            if (index(uppercase(val(1:min(length, len(val)))), 'WINDOWS') > 0) then
                dev = 'NUL'
                return
            end if
        end if

        call get_environment_variable('WINDIR', val, length, stat)
        if (stat == 0 .and. length > 0) then
            dev = 'NUL'
            return
        end if

        dev = '/dev/null'
    end function null_device

end module pyquick
